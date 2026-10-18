"""Pure-numpy feature engineering. No MediaPipe / OpenCV imports here so it is easy to test.

Per-frame feature vector (FEATURE_DIM = 144):
  [0:63]    left-hand shape   (21 landmarks x 3, wrist-relative, scaled by hand size)
  [63:126]  right-hand shape
  [126:138] body block: shoulders, elbows, wrists (6 pts x 2), relative to shoulder centre / shoulder width
  [138:142] hand wrist positions (left x,y ; right x,y) relative to shoulder centre / shoulder width
  [142:144] presence flags (left, right)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

T = 30                       # frames per resampled sequence
N_HAND = 21
FEATURE_DIM = 2 * N_HAND * 3 + 12 + 4 + 2   # 144
POSE_IDX = (11, 12, 13, 14, 15, 16)
IDX_WRISTS = slice(138, 142)
IDX_FLAGS = slice(142, 144)
NOISE_DIMS = 142             # do not add noise to the flag columns
RAISE_Y = 1.8                # hand counts as "raised/active" if wrist is above this (shoulder-widths below shoulders)


@dataclass
class RawFrame:
    left: Optional[np.ndarray] = None     # (21,3) normalised image coords (x,y in 0..1, z relative)
    right: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None     # (33,3)
    aspect: float = 1.0                   # width / height of the source frame


def _hand_shape(h, aspect):
    if h is None:
        return np.zeros(N_HAND * 3, np.float32)
    p = np.asarray(h, np.float64).copy()
    p[:, 0] *= aspect
    p -= p[0]
    size = np.linalg.norm(p[9, :2])
    if size < 1e-6:
        return np.zeros(N_HAND * 3, np.float32)
    return (p / size).ravel().astype(np.float32)


def _body_block(raw: RawFrame):
    body = np.zeros(12, np.float32)
    wr = np.zeros(4, np.float32)
    if raw.pose is None:
        return body, wr
    p = np.asarray(raw.pose, np.float64)[list(POSE_IDX), :2].copy()
    p[:, 0] *= raw.aspect
    center = (p[0] + p[1]) / 2.0
    scale = np.linalg.norm(p[0] - p[1])
    if scale < 1e-6:
        return body, wr
    body = ((p - center) / scale).ravel().astype(np.float32)
    for i, h in enumerate((raw.left, raw.right)):
        if h is not None:
            w = np.asarray(h, np.float64)[0, :2].copy()
            w[0] *= raw.aspect
            wr[2 * i:2 * i + 2] = (w - center) / scale
    return body, wr


def frame_features(raw: RawFrame) -> np.ndarray:
    body, wr = _body_block(raw)
    flags = np.array([raw.left is not None, raw.right is not None], np.float32)
    f = np.concatenate([_hand_shape(raw.left, raw.aspect), _hand_shape(raw.right, raw.aspect), body, wr, flags])
    assert f.shape[0] == FEATURE_DIM
    return f.astype(np.float32)


def active_mask(feats) -> np.ndarray:
    """Boolean per frame: is at least one hand present AND raised into signing space?"""
    f = np.atleast_2d(np.asarray(feats))
    left = (f[:, 142] > 0.5) & (f[:, 139] < RAISE_Y)
    right = (f[:, 143] > 0.5) & (f[:, 141] < RAISE_Y)
    return left | right


def resample_sequence(feats, times, n_out: int = T) -> np.ndarray:
    """Nearest-frame resampling over the span of `times` -> (n_out, D). Speed-invariant."""
    feats = np.asarray(feats, np.float32)
    times = np.asarray(times, np.float64)
    if len(feats) == 0:
        raise ValueError("empty sequence")
    t0, t1 = times[0], times[-1]
    if t1 <= t0:
        idx = np.zeros(n_out, dtype=int)
    else:
        targets = np.linspace(t0, t1, n_out)
        idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
    return feats[idx]


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap augmentation: speed/offset jitter + gaussian noise (flags untouched)."""
    n = len(seq)
    speed = rng.uniform(0.85, 1.15)
    offset = rng.uniform(-2, 2)
    pos = np.clip(np.linspace(0, n - 1, n) * speed + offset, 0, n - 1)
    out = seq[np.round(pos).astype(int)].copy()
    out[:, :NOISE_DIMS] += rng.normal(0, 0.02, size=(n, NOISE_DIMS)).astype(np.float32)
    return out


def vectorize(seqs) -> np.ndarray:
    """(N,T,D) -> (N, T*D + D): flattened sequence + per-feature motion energy."""
    s = np.asarray(seqs, np.float32)
    flat = s.reshape(len(s), -1)
    vel = np.diff(s, axis=1).std(axis=1)
    return np.concatenate([flat, vel], axis=1)
