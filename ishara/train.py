"""Train + honestly evaluate.  python train.py --data data --out models/ishara.joblib"""
import argparse, json, os
import numpy as np
from ishara.dataset import load_dataset
from ishara.model import IsharaModel, cross_validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="models/ishara.joblib")
    ap.add_argument("--kind", default="lr", choices=["lr", "mlp", "rf"])
    ap.add_argument("--aug", type=int, default=4)
    ap.add_argument("--report-dir", default="reports")
    a = ap.parse_args()
    seqs, labels, groups = load_dataset(a.data)
    print(f"{len(seqs)} samples, {len(set(labels))} classes, {len(set(groups))} sessions")
    res = cross_validate(seqs, labels, groups, a.kind, a.aug)
    print(f"\nProtocol : {res['protocol']}\nAccuracy : {res['accuracy']:.3f}\nMacro-F1 : {res['macro_f1']:.3f}\n")
    print(res["report"])
    os.makedirs(a.report_dir, exist_ok=True)
    json.dump({k: v for k, v in res.items() if k != "report"}, open(os.path.join(a.report_dir, "cv.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cm = np.array(res["confusion"]); n = len(res["labels"])
        fig, ax = plt.subplots(figsize=(max(5, n * 0.6), max(4, n * 0.5)))
        ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(res["labels"], rotation=90); ax.set_yticklabels(res["labels"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout()
        fig.savefig(os.path.join(a.report_dir, "confusion.png"), dpi=150)
    except Exception as e:
        print("confusion plot skipped:", e)
    final = IsharaModel(a.kind, a.aug).fit(seqs, labels)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    final.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
