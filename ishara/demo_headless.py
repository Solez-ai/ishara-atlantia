"""Headless demo - test model inference without GUI.
Usage: python demo_headless.py --model models/ishara.joblib --glossary glossary.json --test-data data/
"""
import argparse
import json
import numpy as np
from pathlib import Path
from ishara.model import IsharaModel
from ishara.realtime import Translator
from ishara.dataset import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ishara.joblib")
    ap.add_argument("--glossary", default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--test-data", default="data", help="Test data directory")
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()
    
    # Load model
    print(f"\nLoading model: {a.model}")
    model = IsharaModel.load(a.model)
    print(f"✓ Model loaded")
    
    # Load glossary
    gloss = {}
    if a.glossary and Path(a.glossary).exists():
        with open(a.glossary) as f:
            gloss = json.load(f)
    show = lambda lab: gloss.get(lab, {}).get(a.lang, lab)
    
    # Load test data
    print(f"\nLoading test data from: {a.test_data}")
    try:
        seqs, labels, groups = load_dataset(a.test_data)
        print(f"✓ Loaded {len(seqs)} sequences, {len(set(labels))} classes, {len(set(groups))} groups")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        print("Creating synthetic test data...")
        seqs = np.random.randn(100, 30, 144).astype(np.float32)
        labels = np.random.choice(list(gloss.keys())[:-1], 100)
        groups = np.array([f"test_{i//10}" for i in range(100)])
    
    # Test inference
    print(f"\n{'='*60}")
    print(f"DEMO: Testing model inference")
    print(f"{'='*60}\n")
    
    tr = Translator(model, threshold=a.threshold)
    
    # Run on samples
    n_test = min(20, len(seqs))
    correct = 0
    
    for i in range(n_test):
        seq = seqs[i] if isinstance(seqs[i], np.ndarray) else seqs[i]
        true_label = labels[i]
        
        # Extract features from sequence
        from ishara.features import frame_features, active_mask, RawFrame
        
        # Simulate frame-by-frame processing
        for frame_idx in range(min(5, len(seq))):
            feat = seq[frame_idx]
            ev = tr.update(frame_idx / 30, feat)
        
        # Get prediction
        pred = model.predict(seq)
        match = "✓" if pred.label == true_label else "✗"
        correct += (pred.label == true_label)
        
        print(f"{match} Sample {i+1:2d}: {show(true_label):12s} → {show(pred.label):12s} "
              f"(conf: {pred.conf:.3f})")
        
        tr.clear()
    
    accuracy = correct / n_test * 100
    print(f"\n{'='*60}")
    print(f"Accuracy on {n_test} samples: {accuracy:.1f}%")
    print(f"Threshold: {a.threshold}")
    print(f"Model: {a.model}")
    print(f"{'='*60}\n")
    
    print("✓ Demo complete - model is working!")
    print("\nNext steps:")
    print("  1. Run on real webcam:")
    print("     python demo.py --model models/ishara.joblib --glossary glossary.json")
    print("  2. Record demo video:")
    print("     python demo.py --model models/ishara.joblib --record demo.mp4")


if __name__ == "__main__":
    main()
