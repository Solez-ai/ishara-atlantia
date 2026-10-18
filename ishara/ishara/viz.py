"""Drawing helpers (OpenCV + PIL). PIL is used for text so Bangla can render if a Bengali font is given."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
             (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
POSE_CONN = [(11,12),(11,13),(13,15),(12,14),(14,16)]
LEFT_COL, RIGHT_COL, BODY_COL = (255, 200, 0), (200, 0, 255), (120, 255, 120)   # BGR


def _pt(p, w, h, mirror):
    x = 1 - p[0] if mirror else p[0]
    return int(x * w), int(p[1] * h)


def draw_skeleton(frame, raw, mirror=True):
    h, w = frame.shape[:2]
    if raw.pose is not None:
        for a, b in POSE_CONN:
            cv2.line(frame, _pt(raw.pose[a], w, h, mirror), _pt(raw.pose[b], w, h, mirror), BODY_COL, 3)
        for i in (11, 12, 13, 14, 15, 16):
            cv2.circle(frame, _pt(raw.pose[i], w, h, mirror), 5, BODY_COL, -1)
    for hand, col in ((raw.left, LEFT_COL), (raw.right, RIGHT_COL)):
        if hand is None:
            continue
        for a, b in HAND_CONN:
            cv2.line(frame, _pt(hand[a], w, h, mirror), _pt(hand[b], w, h, mirror), col, 2)
        for p in hand:
            cv2.circle(frame, _pt(p, w, h, mirror), 3, (255, 255, 255), -1)
    return frame


class TextRenderer:
    def __init__(self, font_path=None):
        self.font_path, self._cache = font_path, {}

    def font(self, size):
        if size not in self._cache:
            f = None
            for p in ([self.font_path] if self.font_path else []) + [
                    "DejaVuSans.ttf", "arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
                try:
                    f = ImageFont.truetype(p, size)
                    break
                except Exception:
                    continue
            self._cache[size] = f or ImageFont.load_default()
        return self._cache[size]

    def draw(self, frame, items):
        """items: list of (text, (x, y), size, (r, g, b)). One PIL round-trip per frame."""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, xy, size, col in items:
            d.text(xy, text, font=self.font(size), fill=col)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
