"""Live demo.  python demo.py --model models/ishara.joblib
Optional: --glossary glossary.json  ({"water": {"en": "Water", "bn": "পানি"}})  --font NotoSansBengali.ttf
          --lang bn   --source video.mp4   --record out.mp4
Keys: q quit | c clear sentence | u undo last word"""
import argparse, json, time
import numpy as np
import cv2
from ishara.landmarks import LandmarkExtractor
from ishara.features import frame_features
from ishara.model import IsharaModel
from ishara.realtime import Translator
from ishara.viz import draw_skeleton, TextRenderer

PANEL_W, VIEW_H = 380, 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ishara.joblib")
    ap.add_argument("--source", default="0")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--glossary", default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--font", default=None)
    ap.add_argument("--record", default=None)
    ap.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    a = ap.parse_args()
    gloss = json.load(open(a.glossary, encoding="utf-8")) if a.glossary else {}
    show = lambda lab: gloss.get(lab, {}).get(a.lang, lab)
    tr = Translator(IsharaModel.load(a.model), threshold=a.threshold)
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    ext, txt, writer = LandmarkExtractor(), TextRenderer(a.font), None
    t0, flash_until, flash_word, fps_t, fps = time.time(), 0.0, "", time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time() - t0
        raw = ext.process(frame, t)
        ev = tr.update(t, frame_features(raw))
        if ev["committed"]:
            flash_word, flash_until = show(ev["committed"]), t + 1.2
        view = draw_skeleton(cv2.flip(frame, 1), raw, mirror=True)
        scale = VIEW_H / view.shape[0]
        view = cv2.resize(view, (int(view.shape[1] * scale), VIEW_H))
        panel = np.full((VIEW_H, PANEL_W, 3), (30, 24, 20), np.uint8)
        canvas = np.hstack([view, panel]); x0 = view.shape[1] + 18
        items = [("ISHARA", (x0, 14), 34, (255, 255, 255)),
                 ("sign -> text, on-device", (x0, 56), 18, (170, 170, 170))]
        pred = ev["provisional"] or tr.last_final
        state = "Signing..." if ev["state"] == "signing" else "Raise a hand to sign"
        items.append((state, (x0, 100), 22, (255, 220, 0) if ev["state"] == "signing" else (150, 150, 150)))
        if pred:
            items.append((show(pred.label), (x0, 132), 40, (120, 255, 160)))
            for i, (lab, p) in enumerate(pred.top):
                y = 200 + i * 34
                items.append((f"{show(lab)}", (x0, y), 18, (230, 230, 230)))
                cv2.rectangle(canvas, (x0 + 150, y + 4), (x0 + 150 + int(170 * p), y + 20), (120, 255, 160) if i == 0 else (110, 110, 110), -1)
        sentence = " ".join(show(w) for w in tr.sentence[-8:]) or "..."
        items += [("SENTENCE", (x0, 350), 18, (170, 170, 170)), (sentence, (x0, 376), 28, (255, 255, 255))]
        if t < flash_until:
            items.append((f"+ {flash_word}", (x0, 470), 34, (255, 200, 60)))
        fps = 0.9 * fps + 0.1 / max(time.time() - fps_t, 1e-6); fps_t = time.time()
        items.append((f"{fps:4.0f} fps | CPU only | {ext.backend}", (x0, VIEW_H - 30), 16, (140, 140, 140)))
        canvas = txt.draw(canvas, items)
        if a.record:
            if writer is None:
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), 25, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
        cv2.imshow("Ishara", canvas)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("c"):
            tr.clear()
        elif k == ord("u"):
            tr.undo()
    cap.release(); ext.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
