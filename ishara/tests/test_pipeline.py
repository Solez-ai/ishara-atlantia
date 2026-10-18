"""Hardware-free tests (no webcam, no MediaPipe models). Run: python -m tests.test_pipeline"""
import numpy as np
from ishara.features import (RawFrame, frame_features, FEATURE_DIM, T, resample_sequence,
                             active_mask, augment_sequence, vectorize)
from ishara.model import IsharaModel, cross_validate
from ishara.realtime import Translator, EpisodeSegmenter
from ishara.landmarks import assign_hands


def synth_template(rng, n_classes):
    """One smooth random trajectory per class in feature space (flags on, wrists 'raised')."""
    temps = []
    for _ in range(n_classes):
        a = rng.normal(0, 1, FEATURE_DIM); b = rng.normal(0, 1, FEATURE_DIM)
        temps.append((a, b))
    return temps


def synth_seq(temp, rng, n=T, noise=0.15, shift=0.0):
    u = np.linspace(0, 1, n)[:, None]
    seq = temp[0][None] * np.sin(np.pi * (u + shift)) + temp[1][None] * u
    seq = seq + rng.normal(0, noise, seq.shape)
    seq[:, 142:144] = 1.0
    seq[:, 138:142] = 0.3
    return seq.astype(np.float32)


def make_data(n_classes=8, sessions=4, per=10, seed=0):
    rng = np.random.default_rng(seed)
    temps = synth_template(rng, n_classes)
    X, y, g = [], [], []
    for c in range(n_classes):
        for s in range(sessions):
            off = rng.normal(0, 0.1, FEATURE_DIM)
            for _ in range(per):
                X.append(synth_seq(temps[c], rng) + off.astype(np.float32) * np.r_[np.ones(142), [0, 0]].astype(np.float32))
                y.append(f"sign{c}"); g.append(f"s{s}")
    return np.stack(X), np.array(y), np.array(g), temps


def test_features_shape_and_missing():
    rng = np.random.default_rng(0)
    hand = rng.random((21, 3)).astype(np.float32)
    pose = rng.random((33, 3)).astype(np.float32)
    f = frame_features(RawFrame(left=hand, right=None, pose=pose, aspect=1.78))
    assert f.shape == (FEATURE_DIM,) and f[142] == 1 and f[143] == 0
    assert np.all(f[63:126] == 0)                       # missing right hand -> zeros
    f2 = frame_features(RawFrame())                      # nothing detected
    assert f2.shape == (FEATURE_DIM,) and not active_mask(f2)[0]
    assert np.isfinite(f).all() and np.isfinite(f2).all()


def test_resample_speed_invariance():
    n = 47
    feats = np.tile(np.linspace(0, 1, n)[:, None], (1, FEATURE_DIM)).astype(np.float32)
    a = resample_sequence(feats, np.linspace(0, 1.0, n))
    b = resample_sequence(feats, np.linspace(0, 3.0, n))       # same motion, 3x slower
    assert a.shape == (T, FEATURE_DIM) and np.allclose(a, b)


def test_augment_keeps_flags():
    rng = np.random.default_rng(1)
    seq = synth_seq(synth_template(rng, 1)[0], rng)
    out = augment_sequence(seq, rng)
    assert out.shape == seq.shape and np.all(out[:, 142:144] == 1.0)


def test_train_and_session_held_out_cv():
    X, y, g, _ = make_data()
    res = cross_validate(X, y, g, kind="lr", aug=2)
    assert "session-held-out" in res["protocol"]
    assert res["accuracy"] > 0.9, res["accuracy"]
    m = IsharaModel("lr", 2).fit(X, y)
    p = m.predict(X[0])
    assert p.label == y[0] and 0 <= p.conf <= 1 and len(p.top) == 3


def test_translator_commits_one_word_per_sign():
    X, y, g, temps = make_data(n_classes=5, sessions=3, per=8, seed=3)
    m = IsharaModel("lr", 2).fit(X, y)
    tr = Translator(m, threshold=0.5)
    rng = np.random.default_rng(9)
    idle = np.zeros(FEATURE_DIM, np.float32)
    t, dt, committed = 0.0, 1 / 30, []
    def run(seconds, feat_fn):
        nonlocal t
        for i in range(int(seconds * 30)):
            ev = tr.update(t, feat_fn(i)); t += dt
            if ev["committed"]:
                committed.append(ev["committed"])
    run(1.0, lambda i: idle)
    for cls in (2, 4):                                         # two signs, hands down between
        n = 45
        seq = synth_seq(temps[cls], rng, n=n)
        run(n / 30, lambda i, s=seq: s[min(i, n - 1)])
        run(1.0, lambda i: idle)
    assert committed == ["sign2", "sign4"], committed
    assert tr.sentence == ["sign2", "sign4"]


def test_ignored_label_not_committed():
    X, y, g, temps = make_data(n_classes=3, sessions=3, per=8, seed=5)
    y = np.array([("_other" if l == "sign0" else l) for l in y])
    m = IsharaModel("lr", 2).fit(X, y)
    tr = Translator(m, threshold=0.5)
    rng = np.random.default_rng(2)
    seq = synth_seq(temps[0], rng, n=40)
    t = 0.0
    for i in range(40):
        tr.update(t, seq[i]); t += 1 / 30
    for _ in range(30):
        tr.update(t, np.zeros(FEATURE_DIM, np.float32)); t += 1 / 30
    assert tr.last_final.label == "_other" and tr.sentence == []


def test_short_flicker_is_ignored():
    seg = EpisodeSegmenter()
    f = np.zeros(FEATURE_DIM, np.float32); f[142] = 1
    out = [seg.push(i / 30, f) for i in range(4)] + [seg.push(0.2 + i / 30, np.zeros(FEATURE_DIM, np.float32)) for i in range(30)]
    assert all(o is None for o in out)


def test_assign_hands():
    pose = np.zeros((33, 3), np.float32); pose[15, :2] = (0.7, 0.5); pose[16, :2] = (0.3, 0.5)
    h_near_left = np.zeros((21, 3), np.float32); h_near_left[0, :2] = (0.69, 0.5)
    l, r = assign_hands([h_near_left], pose)
    assert l is not None and r is None
    h_right = np.zeros((21, 3), np.float32); h_right[0, :2] = (0.31, 0.5)
    l, r = assign_hands([h_right, h_near_left], pose)
    assert l is h_near_left and r is h_right


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
