"""Small scikit-learn classifier over resampled landmark sequences. Trains in seconds on a CPU."""
from dataclasses import dataclass
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .features import vectorize, augment_sequence, T, FEATURE_DIM


@dataclass
class Prediction:
    label: str
    conf: float
    top: list          # [(label, prob), ...] best first


def _make_pipeline(kind, n_samples, n_dims):
    n_comp = int(max(2, min(128, n_samples - 1, n_dims)))
    if kind == "lr":
        clf = LogisticRegression(C=1.0, max_iter=3000)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=500, random_state=0)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    else:
        raise ValueError(f"unknown kind {kind}")
    return Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=0)), ("clf", clf)])


def augment_many(seqs, labels, k, rng):
    if k <= 0:
        return seqs, labels
    xs, ys = [seqs], [labels]
    for _ in range(k):
        xs.append(np.stack([augment_sequence(s, rng) for s in seqs]))
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


class IsharaModel:
    def __init__(self, kind="lr", aug=4, seed=0):
        self.kind, self.aug, self.seed = kind, aug, seed
        self.pipe = None
        self.classes_ = None

    def fit(self, seqs, labels):
        rng = np.random.default_rng(self.seed)
        X, y = augment_many(np.asarray(seqs), np.asarray(labels), self.aug, rng)
        Xv = vectorize(X)
        self.pipe = _make_pipeline(self.kind, len(Xv), Xv.shape[1])
        self.pipe.fit(Xv, y)
        self.classes_ = list(self.pipe.classes_)
        return self

    def predict_proba(self, seqs):
        return self.pipe.predict_proba(vectorize(np.asarray(seqs)))

    def predict(self, seq) -> Prediction:
        p = self.predict_proba(np.asarray(seq)[None])[0]
        order = np.argsort(p)[::-1]
        top = [(self.classes_[i], float(p[i])) for i in order[:3]]
        return Prediction(top[0][0], top[0][1], top)

    def save(self, path):
        joblib.dump({"model": self, "T": T, "dim": FEATURE_DIM}, path)

    @staticmethod
    def load(path):
        blob = joblib.load(path)
        assert blob["T"] == T and blob["dim"] == FEATURE_DIM, "model/feature version mismatch"
        return blob["model"]


def cross_validate(seqs, labels, groups, kind="lr", aug=4, seed=0):
    """Session-held-out CV when >=2 sessions exist (honest); otherwise stratified CV (optimistic)."""
    seqs, labels, groups = np.asarray(seqs), np.asarray(labels), np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) >= 2:
        splitter = GroupKFold(n_splits=min(5, len(uniq)))
        splits = list(splitter.split(seqs, labels, groups))
        protocol = f"GroupKFold over {len(uniq)} sessions (session-held-out)"
    else:
        n = int(min(5, np.min(np.unique(labels, return_counts=True)[1])))
        splits = list(StratifiedKFold(n_splits=max(2, n), shuffle=True, random_state=seed).split(seqs, labels))
        protocol = "StratifiedKFold (only ONE session: optimistic, record more sessions/signers)"
    y_true, y_pred = [], []
    for tr, te in splits:
        m = IsharaModel(kind, aug, seed).fit(seqs[tr], labels[tr])
        y_pred.extend([m.predict(s).label for s in seqs[te]])
        y_true.extend(labels[te])
    names = sorted(set(labels))
    return {
        "protocol": protocol,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "labels": names,
        "confusion": confusion_matrix(y_true, y_pred, labels=names).tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
