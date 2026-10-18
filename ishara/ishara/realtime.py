"""Streaming logic: hands up -> sign -> hands down -> one word. Pure Python/numpy, fully testable."""
from typing import Optional
import numpy as np
from .features import active_mask, resample_sequence, T
from .model import Prediction

IGNORE_PREFIX = "_"     # labels like "_other" are recognised but never added to the sentence


class EpisodeSegmenter:
    """Turns a frame stream into 'episodes' = spans where a hand is raised. A gap shorter than
    `gap_tol` seconds is bridged, so brief tracking dropouts do not split a sign."""

    def __init__(self, gap_tol=0.4, min_dur=0.35, max_dur=6.0):
        self.gap_tol, self.min_dur, self.max_dur = gap_tol, min_dur, max_dur
        self.reset()

    def reset(self):
        self.f, self.t, self.in_episode, self.last_active = [], [], False, None

    def current(self):
        if len(self.f) < 2:
            return None
        return np.stack(self.f), np.array(self.t)

    def push(self, t, feat):
        """Returns (feats, times) when an episode has just finished, else None."""
        act = bool(active_mask(feat)[0])
        if act:
            if not self.in_episode:
                self.in_episode, self.f, self.t = True, [], []
            self.f.append(np.asarray(feat, np.float32))
            self.t.append(t)
            self.last_active = t
        if not self.in_episode:
            return None
        too_long = (t - self.t[0]) > self.max_dur
        gap_over = (not act) and (t - self.last_active) >= self.gap_tol
        if gap_over or too_long:
            out = None
            if len(self.f) >= 3 and (self.t[-1] - self.t[0]) >= self.min_dur:
                out = (np.stack(self.f), np.array(self.t))
            self.reset()
            return out
        return None


class Translator:
    def __init__(self, model, threshold=0.55, live_interval=0.4, min_live_dur=0.5):
        self.model, self.threshold = model, threshold
        self.live_interval, self.min_live_dur = live_interval, min_live_dur
        self.seg = EpisodeSegmenter()
        self.sentence = []
        self.provisional: Optional[Prediction] = None
        self.last_final: Optional[Prediction] = None
        self._last_live = -1e9

    def clear(self):
        self.sentence.clear()

    def undo(self):
        if self.sentence:
            self.sentence.pop()

    def _predict(self, feats, times):
        return self.model.predict(resample_sequence(feats, times, T))

    def update(self, t, feat):
        """Feed one frame. Returns dict(state, provisional, final, committed)."""
        finished = self.seg.push(t, feat)
        committed, final = None, None
        if finished is not None:
            final = self._predict(*finished)
            self.last_final = final
            self.provisional = None
            if final.conf >= self.threshold and not final.label.startswith(IGNORE_PREFIX):
                committed = final.label
                self.sentence.append(committed)
        elif self.seg.in_episode:
            cur = self.seg.current()
            if cur is not None and (cur[1][-1] - cur[1][0]) >= self.min_live_dur and (t - self._last_live) >= self.live_interval:
                self.provisional = self._predict(*cur)
                self._last_live = t
        else:
            self.provisional = None
        return {"state": "signing" if self.seg.in_episode else "idle",
                "provisional": self.provisional, "final": final, "committed": committed}
