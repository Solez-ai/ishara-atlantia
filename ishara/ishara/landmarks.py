"""MediaPipe wrapper. Supports both the legacy `mp.solutions.holistic` API (mediapipe <= ~0.10.14)
and the newer Tasks API (HandLandmarker + PoseLandmarker, needs two small model files that are
downloaded once into ./models). Frames are processed UN-mirrored; mirror only for display."""
import os, urllib.request
import numpy as np
import cv2
from .features import RawFrame

HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def _arr(lms):
    return np.array([[l.x, l.y, l.z] for l in lms], np.float32)


def _ensure(url, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, os.path.basename(url))
    if not os.path.exists(path):
        print(f"Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            raise RuntimeError(f"Could not download {url}: {e}\nDownload it manually into {model_dir}/") from e
    return path


def assign_hands(hands, pose):
    """Map a list of detected hands to (left, right) slots using the pose wrists; fall back to image x."""
    if not hands:
        return None, None
    if pose is not None:
        lw, rw = pose[15, :2], pose[16, :2]
        if len(hands) == 1:
            dl = np.linalg.norm(hands[0][0, :2] - lw)
            dr = np.linalg.norm(hands[0][0, :2] - rw)
            return (hands[0], None) if dl <= dr else (None, hands[0])
        a, b = hands[0], hands[1]
        c_ab = np.linalg.norm(a[0, :2] - lw) + np.linalg.norm(b[0, :2] - rw)
        c_ba = np.linalg.norm(b[0, :2] - lw) + np.linalg.norm(a[0, :2] - rw)
        return (a, b) if c_ab <= c_ba else (b, a)
    hs = sorted(hands, key=lambda h: h[0, 0])
    if len(hs) == 1:
        return (hs[0], None) if hs[0][0, 0] < 0.5 else (None, hs[0])
    return hs[0], hs[-1]


class LandmarkExtractor:
    def __init__(self, model_dir="models", prefer="auto"):
        import mediapipe as mp
        self.mp = mp
        self._last_ms = -1
        if prefer in ("auto", "legacy") and hasattr(mp, "solutions"):
            self.backend = "legacy"
            self._h = mp.solutions.holistic.Holistic(
                model_complexity=1, smooth_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
            self.backend = "tasks"
            hp, pp = _ensure(HAND_URL, model_dir), _ensure(POSE_URL, model_dir)
            self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=hp),
                running_mode=vision.RunningMode.VIDEO, num_hands=2))
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=pp),
                running_mode=vision.RunningMode.VIDEO))

    def process(self, bgr, t_seconds) -> RawFrame:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "legacy":
            rgb.flags.writeable = False
            r = self._h.process(rgb)
            hands = [_arr(x.landmark) for x in (r.left_hand_landmarks, r.right_hand_landmarks) if x is not None]
            pose = _arr(r.pose_landmarks.landmark) if r.pose_landmarks else None
        else:
            ms = max(int(t_seconds * 1000), self._last_ms + 1)   # timestamps must strictly increase
            self._last_ms = ms
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            hr = self._hand.detect_for_video(img, ms)
            pr = self._pose.detect_for_video(img, ms)
            hands = [_arr(x) for x in hr.hand_landmarks]
            pose = _arr(pr.pose_landmarks[0]) if pr.pose_landmarks else None
        left, right = assign_hands(hands, pose)
        return RawFrame(left, right, pose, w / h)

    def close(self):
        try:
            (self._h if self.backend == "legacy" else self._hand).close()
        except Exception:
            pass
