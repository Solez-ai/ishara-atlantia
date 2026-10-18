"""Sample storage. Layout: data/<label>/<session>__<n>.npy, each file is a (T, FEATURE_DIM) float32 array.
`session` should identify signer + recording day (e.g. "amina_d1") so evaluation can hold out whole sessions."""
import os, glob
import numpy as np
from .features import T, FEATURE_DIM


def save_sample(data_dir, label, session, seq):
    d = os.path.join(data_dir, label)
    os.makedirs(d, exist_ok=True)
    n = len(glob.glob(os.path.join(d, f"{session}__*.npy")))
    path = os.path.join(d, f"{session}__{n:04d}.npy")
    np.save(path, np.asarray(seq, np.float32))
    return path


def delete_last(data_dir, label, session):
    files = sorted(glob.glob(os.path.join(data_dir, label, f"{session}__*.npy")))
    if files:
        os.remove(files[-1])
        return files[-1]
    return None


def count_samples(data_dir, label):
    return len(glob.glob(os.path.join(data_dir, label, "*.npy")))


def load_dataset(data_dir):
    seqs, labels, groups = [], [], []
    for label in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, label)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            a = np.load(f)
            if a.shape != (T, FEATURE_DIM):
                print(f"skip {f}: bad shape {a.shape}")
                continue
            seqs.append(a)
            labels.append(label)
            groups.append(os.path.basename(f).split("__")[0])
    if not seqs:
        raise RuntimeError(f"no samples found in {data_dir}")
    return np.stack(seqs), np.array(labels), np.array(groups)
