"""Interactive sign language demo - works without webcam/display.
Simulates real-time signing for testing and demonstration.

Usage: python interactive_demo.py --model models/ishara.joblib --glossary glossary.json --lang bn
"""
import argparse
import json
import numpy as np
from pathlib import Path
from ishara.model import IsharaModel
from ishara.realtime import Translator
from ishara.dataset import load_dataset


def main():
    ap = argparse.ArgumentParser(description="Interactive sign language recognition demo")
    ap.add_argument("--model", default="models/ishara_mlp.joblib", help="Model path")
    ap.add_argument("--glossary", default="glossary.json", help="Glossary JSON")
    ap.add_argument("--lang", default="en", choices=["en", "bn"], help="Display language")
    ap.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold")
    a = ap.parse_args()
    
    print("\n" + "="*70)
    print("ISHARA INTERACTIVE DEMO - Sign Language Recognition")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model: {a.model}")
    try:
        model = IsharaModel.load(a.model)
        print(f"✓ Model loaded\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Load glossary
    gloss = {}
    if Path(a.glossary).exists():
        with open(a.glossary) as f:
            gloss = json.load(f)
    
    show = lambda lab: gloss.get(lab, {}).get(a.lang, lab) if lab in gloss else lab
    
    # Load test data for sampling
    try:
        seqs, labels, groups = load_dataset("data")
        print(f"Loaded {len(seqs)} test samples from data/\n")
    except Exception as e:
        print(f"Warning: Could not load test data: {e}")
        seqs, labels, groups = None, None, None
    
    # Initialize translator
    tr = Translator(model, threshold=a.threshold)
    
    print("COMMANDS:")
    print("  1-20: Sign a word (or 'q' to quit, 'c' to clear)")
    print("  'p':  Show sample predictions (if data available)")
    print("  'c':  Clear sentence")
    print("  'q':  Quit\n")
    
    print(f"Vocabulary ({len(gloss)} classes):")
    signs = [k for k in gloss.keys() if k != "_other"]
    for i, sign in enumerate(signs, 1):
        print(f"  {i:2d}. {sign:12s} → {show(sign):20s} ({gloss[sign]['bn']})")
    print(f"  21. _other      → Non-sign gesture")
    
    print("\n" + "-"*70)
    print("DEMO MODE: Simulating signing from test data")
    print("-"*70 + "\n")
    
    sentence = []
    
    while True:
        try:
            cmd = input("Enter sign number (1-20), 'p' for predictions, 'c' to clear, 'q' to quit: ").strip().lower()
            
            if cmd == 'q':
                print("\n✓ Goodbye!")
                break
            
            elif cmd == 'c':
                sentence = []
                tr.clear()
                print("Sentence cleared.")
                print(f"Current sentence: {' '.join(show(w) for w in sentence) or '(empty)'}\n")
            
            elif cmd == 'p':
                if seqs is not None and len(seqs) > 0:
                    print("\nSample predictions from test data:\n")
                    for i in range(min(5, len(seqs))):
                        idx = np.random.randint(len(seqs))
                        pred = model.predict(seqs[idx])
                        true_label = labels[idx]
                        match = "✓" if pred.label == true_label else "✗"
                        print(f"  {match} True: {show(true_label):15s} | "
                              f"Predicted: {show(pred.label):15s} (conf: {pred.conf:.3f})")
                    print()
                else:
                    print("No test data available\n")
            
            else:
                try:
                    sign_idx = int(cmd) - 1
                    if 0 <= sign_idx < len(signs):
                        sign_name = signs[sign_idx]
                    elif sign_idx == 20:
                        sign_name = "_other"
                    else:
                        print("Invalid number. Enter 1-20 or a command.\n")
                        continue
                    
                    # Get a sample of this sign from test data
                    if seqs is not None:
                        mask = labels == sign_name
                        if np.any(mask):
                            idx = np.where(mask)[0][np.random.randint(np.sum(mask))]
                            sample = seqs[idx]
                        else:
                            # Random sample if sign not found
                            sample = np.random.randn(30, 144).astype(np.float32)
                    else:
                        sample = np.random.randn(30, 144).astype(np.float32)
                    
                    # Simulate frame-by-frame processing
                    print(f"\n[Simulating signing: {sign_name}]")
                    for frame_idx in range(30):
                        feat = sample[frame_idx]
                        t = frame_idx / 30.0
                        ev = tr.update(t, feat)
                        
                        # Show state transitions
                        if ev.get("state") == "signing":
                            print(f"  Frame {frame_idx:2d}: SIGNING... confidence={ev.get('provisional').conf if ev.get('provisional') else 'N/A'}")
                    
                    # Get final prediction
                    pred = model.predict(sample)
                    
                    print(f"\n  Result: {show(pred.label)}")
                    print(f"  Confidence: {pred.conf:.3f}")
                    
                    if pred.conf >= a.threshold:
                        sentence.append(pred.label)
                        print(f"  ✓ COMMITTED\n")
                    else:
                        print(f"  ✗ Below threshold ({a.threshold}), not committed\n")
                    
                    print(f"Sentence so far: {' '.join(show(w) for w in sentence) or '(empty)'}")
                    print(f"Words: {len(sentence)}\n")
                    
                except ValueError:
                    print("Invalid input. Enter number 1-20 or a command.\n")
        
        except KeyboardInterrupt:
            print("\n\n✓ Demo interrupted by user")
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n" + "="*70)
    print(f"FINAL SENTENCE: {' '.join(show(w) for w in sentence) or '(empty)'}")
    print(f"Total words recognized: {len(sentence)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
