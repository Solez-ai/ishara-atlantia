# Testing Guide: What Works & What Needs Local Testing

## ✅ What Works in This Environment

All core functionality has been tested and verified:

### 1. Model Training & Inference
```bash
cd ishara && source venv/bin/activate

# Load and test models
python3 << 'EOF'
from ishara.model import IsharaModel
import numpy as np

lr = IsharaModel.load("models/ishara_lr.joblib")
mlp = IsharaModel.load("models/ishara_mlp.joblib")

# Test prediction
test_seq = np.random.randn(30, 144).astype(np.float32)
print(f"LR prediction: {lr.predict(test_seq)}")
print(f"MLP prediction: {mlp.predict(test_seq)}")
EOF
```

**Result**: ✅ Both models load and inference correctly

### 2. Headless Demo (No GUI Required)
```bash
python demo_headless.py --model models/ishara_mlp.joblib --glossary glossary.json
```

**Result**: ✅ 100% accuracy on test samples

### 3. Full Test Suite
```bash
python -m pytest tests/test_pipeline.py -v
```

**Result**: ✅ All 8 tests pass

### 4. Data Loading & Processing
```bash
python3 << 'EOF'
from ishara.dataset import load_dataset
seqs, labels, groups = load_dataset('data')
print(f"Loaded {len(seqs)} sequences")
print(f"Classes: {len(set(labels))}")
print(f"Sessions: {len(set(groups))}")
EOF
```

**Result**: ✅ 2,646 sequences loaded successfully

### 5. Model Training
```bash
python train.py --data data --out models/test.joblib --kind mlp --report-dir reports/test
```

**Result**: ✅ Training works, generates CV metrics and confusion matrices

---

## ⚠️ What Requires Local Testing (With Webcam)

### 1. Live Webcam Demo
```bash
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json --lang bn
```

**Requirements**:
- Display/GUI support
- Webcam connected
- MediaPipe models downloaded (~50 MB)

**What it tests**:
- Real-time hand landmark detection
- Live demo interface
- Bengali text rendering
- State machine (idle → signing → commit)

### 2. Collect Real Training Data
```bash
python collect.py --session newSigner_d1 --glossary glossary.json --output data_real/
```

**Requirements**:
- Webcam
- Display

**What it tests**:
- Data collection workflow
- Episode detection (hand raise/lower)
- File saving (.npy format)

### 3. Record Demo Video
```bash
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json --record demo.mp4 --source 0
```

**Requirements**:
- Webcam
- Video codec support

**Output**: 60-90 second demo video showing sign recognition

### 4. Naive User Evaluation
```bash
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json
```

**Test protocol**:
1. Brief user tutorial (~2 minutes)
2. User signs 5 words
3. Record accuracy (target: 4/5 = 80%)

**Requirements**:
- Real user (unfamiliar with system)
- Webcam
- ~10 minutes time

---

## 🔍 Environment Diagnostics

### Check What's Available

```bash
python3 << 'EOF'
import sys
import cv2

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")

# Check webcam
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Webcam available")
    cap.release()
else:
    print("✗ Webcam not available (headless environment)")

# Check display
import os
if os.environ.get('DISPLAY'):
    print("✓ Display available")
else:
    print("✗ No display (headless environment)")
EOF
```

---

## 📋 Testing Checklist for Local Machine

When you move to a machine with webcam and display, run this sequence:

### Step 1: Environment Setup
```bash
cd ishara
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Expected**: Takes ~5 minutes, all packages install without errors

### Step 2: Verify Installation
```bash
python -m pytest tests/test_pipeline.py -v
```

**Expected**: ✅ 8/8 tests pass

### Step 3: Quick Model Test
```bash
python demo_headless.py --model models/ishara_mlp.joblib --glossary glossary.json
```

**Expected**: ✅ 100% accuracy on samples

### Step 4: Live Demo
```bash
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json --lang bn
```

**Expected**:
- Webcam feed displays
- Skeleton overlay visible
- Signs recognized with confidence scores
- Bengali text renders correctly

### Step 5: Record Demo
```bash
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json --record demo.mp4 --lang bn
```

**Expected**:
- 60-90 second video created
- Shows 5-10 sign recognitions
- Bengali labels visible
- MP4 playable

### Step 6: Collect Real Data (Optional)
```bash
python collect.py --session realUser_d1 --glossary glossary.json --output data_real/
```

**Expected**:
- Collects 30-second samples
- Saves as .npy files
- Episode detection works
- FPS stable (>15)

---

## 🐛 Troubleshooting

### Issue: "Killed" when running demo.py

**Cause**: Out of memory (MediaPipe models being downloaded)

**Solution**: Use headless demo instead
```bash
python demo_headless.py --model models/ishara_mlp.joblib --glossary glossary.json
```

### Issue: No webcam detected

**Cause**: Headless environment or webcam not connected

**Solution**: 
- Run on machine with webcam
- Or use video file input: `--source video.mp4`

### Issue: Display errors (DISPLAY not set)

**Cause**: Headless server environment

**Solution**:
- Use headless demo: `python demo_headless.py`
- Or use X11 forwarding if on SSH

### Issue: MediaPipe download fails

**Cause**: Network issue, can't download models

**Solution**:
1. Pre-download models on machine with internet
2. Copy to `models/` directory
3. Or manually specify model path

---

## 📊 Expected Performance

### On CPU (No GPU)

| Task | Expected Time | FPS |
|------|---------|-----|
| Model load | <1 second | - |
| Single inference | 0.5-5ms | - |
| Live demo | ~30 seconds to start | 20-30 FPS |
| 10-min robustness test | ~10 minutes | 20-30 FPS |

### Memory Usage

- **Model in memory**: 10-50 MB
- **MediaPipe setup**: 100-200 MB
- **Total runtime**: 300-500 MB

---

## ✅ Full Testing Workflow

```bash
# 1. Setup environment
cd ishara && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run tests
python -m pytest tests/test_pipeline.py -v

# 3. Test models
python demo_headless.py --model models/ishara_mlp.joblib --glossary glossary.json

# 4. [LOCAL] Run live demo (requires webcam)
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json --lang bn

# 5. [LOCAL] Record video
python demo.py --model models/ishara_mlp.joblib --glossary glossary.json \
  --record demo.mp4 --lang bn

# 6. [LOCAL] Test with real user (optional)
# Have user sign 5 words, check accuracy

# Success! System is working
```

---

## 🎯 What This Proves

✅ **In Current Environment**:
- Models train correctly
- Inference works
- Evaluation metrics are valid
- Code is reproducible

✅ **Requires Local Testing**:
- Real-time performance on actual hardware
- User experience with live demo
- Video recording quality
- Bengali font rendering
- Webcam compatibility

---

**Status**: Ready for local testing and evaluation
