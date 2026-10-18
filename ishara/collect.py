"""Record training samples with your webcam.
  python collect.py --labels hello,thanks,water,_other --session amina_d1
Keys: SPACE = arm one recording (raise hand, sign, lower hand -> saved automatically)
      n / p = next / previous label     u = undo last sample     q = quit
Only landmarks are stored (no video, no faces)."""
import argparse, time
import cv2
from ishara.landmarks import LandmarkExtractor
from ishara.features import frame_features, resample_sequence, T
from ishara.realtime import EpisodeSegmenter
from ishara.dataset import save_sample, delete_last, count_samples
from ishara.viz import draw_skeleton, TextRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="comma-separated sign names; use _other for non-sign movement")
    ap.add_argument("--session", required=True, help="signer + day id, e.g. amina_d1")
    ap.add_argument("--data", default="data")
    ap.add_argument("--source", default="0")
    ap.add_argument("--font", default=None)
    a = ap.parse_args()
    labels = [s.strip() for s in a.labels.split(",") if s.strip()]
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    ext, seg, txt = LandmarkExtractor(), EpisodeSegmenter(gap_tol=0.5), TextRenderer(a.font)
    idx, armed, msg, t0 = 0, False, "Press SPACE, then sign", time.time()
    armed_at = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        raw = ext.process(frame, t)
        feat = frame_features(raw)
        if armed:
            done = seg.push(t, feat)
            if done is not None:
                path = save_sample(a.data, labels[idx], a.session, resample_sequence(done[0], done[1], T))
                msg, armed = f"saved {path.split('/')[-1]}", False
            elif t - armed_at > 10:
                msg, armed = "timed out - press SPACE again", False
                seg.reset()
        view = draw_skeleton(cv2.flip(frame, 1), raw, mirror=True)
        status = "RECORDING - sign now, then lower your hand" if armed else msg
        items = [(f"Label: {labels[idx]}   ({count_samples(a.data, labels[idx])} samples)", (16, 12), 28, (255, 255, 255)),
                 (status, (16, 52), 24, (255, 220, 0) if armed else (200, 200, 200)),
                 ("SPACE record | n/p label | u undo | q quit", (16, view.shape[0] - 34), 20, (200, 200, 200))]
        cv2.imshow("Ishara - collect", txt.draw(view, items))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            armed, armed_at = True, t
            seg.reset()
        elif k == ord("n"):
            idx = (idx + 1) % len(labels)
        elif k == ord("p"):
            idx = (idx - 1) % len(labels)
        elif k == ord("u"):
            d = delete_last(a.data, labels[idx], a.session)
            msg = f"deleted {d}" if d else "nothing to delete"
    cap.release(); cv2.destroyAllWindows(); ext.close()


if __name__ == "__main__":
    main()
