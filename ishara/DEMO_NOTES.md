# Phase 3: Live Demo and Interface Documentation

## 3.1 State Machine: Idle → Signing → Word Commits

The demo implements a robust state machine for converting continuous hand motion into discrete word recognitions.

### State Transitions

```
    Idle (hands down)
         ↓
    [Hand raised for > 0.3s]
         ↓
    Signing (recognizing frames)
         ↓
    [Hand lowered for > 0.5s]
         ↓
    Word Committed ← [Use top prediction if conf > threshold]
         ↓
    Back to Idle
```

**State codes**:
- `"idle"`: Hands at rest, no sign recognition
- `"signing"`: Hands raised, actively recognizing
- `"committed"`: Sign recognized, word added to sentence

**Parameters**:
- `raise_threshold`: 0.3s (hand must be up to enter signing state)
- `lower_threshold`: 0.5s (hand must be down to commit and exit)
- `confidence_threshold`: 0.85 (minimum confidence to commit word)

### Implementation

State machine in `ishara/realtime.py`:
- `EpisodeSegmenter`: Detects hand raise/lower transitions
- `Translator`: Manages state and converts sign predictions to word commits
- `demo.py`: Visualization layer

## 3.2 Bengali Font Support

### Configuration

The demo supports rendering Bengali text using:
1. **Preferred**: Noto Sans Bengali (Google Fonts)
   - Download: https://fonts.google.com/noto/specimen/Noto+Sans+Bengali
   - Install: Place `NotoSansBengali.ttf` in `fonts/` directory
   - Usage: `python demo.py --font fonts/NotoSansBengali.ttf --lang bn`

2. **Fallback**: System fonts via PIL
   - Automatically used if Noto Sans not available
   - May have limited Bengali character support

### Conjunct Character Rendering

Bengali script uses complex conjunct characters (যুক্তাক্ষর). The text renderer uses PIL + FontTools for proper rendering:

**Example conjuncts**:
- ক্ষ = "ka" + "ksa" ligature
- দ্ধ = "da" + "dha" ligature
- ন্ত = "na" + "ta" ligature

**Verification**:
- Test string: "ঐতিহ্যবাহী" (traditional)
- Expected: Proper character positioning and kerning
- Font engine: PIL/Pillow (handles shaping via fontTools)

## 3.3 Conjunct Character Verification

The glossary includes Bengali words with conjunct characters. Rendering test:

```python
from ishara.viz import TextRenderer
import numpy as np
import cv2

renderer = TextRenderer("fonts/NotoSansBengali.ttf")
test_words = {
    "water": "পানি",      # Simple characters
    "school": "স্কুল",    # Conjunct: ্ক
    "hospital": "হাসপাতাল",  # Complex
    "family": "পরিবার"    # Multiple conjuncts
}

for en, bn in test_words.items():
    canvas = np.zeros((100, 400, 3), np.uint8)
    renderer.draw(canvas, [(bn, (50, 50), 32, (255, 255, 255))])
    # Visually verify proper rendering
```

## 3.4 Optional Text-to-Speech Support

Implement TTS for spoken output of recognized sentences (future enhancement):

### Implementation Plan

1. **Library**: Use `gTTS` (Google Text-to-Speech) or `pyttsx3` (offline)
   ```bash
   pip install gtts pyttsx3
   ```

2. **Integration**:
   ```python
   from pyttsx3 import init as tts_init
   engine = tts_init()
   engine.say("পানি খাই")  # "Water drink"
   engine.runAndWait()
   ```

3. **Demo flag**:
   ```bash
   python demo.py --tts on --tts-lang bn
   ```

### Current Status

- **Core demo**: Fully functional without TTS
- **TTS module**: Optional enhancement
- **Priority**: Low (text display sufficient for MVP)

## 3.5 Video File Input Fallback

Demo supports both live webcam and video file input:

### Usage

**Live webcam (default)**:
```bash
python demo.py --model models/ishara.joblib --glossary glossary.json
```

**Video file**:
```bash
python demo.py --model models/ishara.joblib --glossary glossary.json --source input_video.mp4
```

**Output video**:
```bash
python demo.py --model models/ishara.joblib --glossary glossary.json --source input_video.mp4 --record output.mp4
```

### Video Format Requirements

- **Codec**: H.264, MJPEG, or VP9
- **Resolution**: Any (auto-scaled to 600px height)
- **Frame rate**: 20-30 FPS (works with any FPS, auto-scaled)
- **Audio**: Ignored (optional)

## 3.6 Demo Video Recording

### Specification

- **Duration**: 60-90 seconds
- **Content**: 
  - 5-10 sign demonstrations
  - Clear hand visibility
  - Natural transitions between signs
  - Idle periods (hands at rest)
  - Sentence formation example

### Recording Command

```bash
python demo.py \
  --model models/ishara_mlp.joblib \
  --glossary glossary.json \
  --font fonts/NotoSansBengali.ttf \
  --lang bn \
  --source 0 \
  --record demo.mp4
```

### Demo Script (Recommended)

Sign the following sequence:
1. "hello" (welcome gesture)
2. "water" (hand gesture)
3. "thankyou" (gratitude)
4. Pause and show sentence: "hello water thankyou"
5. "goodbye" (farewell)

**Estimated recording**: ~75 seconds

## 3.7 10-Minute Robustness Test

Verifies system stability during extended use:

### Test Protocol

```bash
timeout 600 python demo.py \
  --model models/ishara_mlp.joblib \
  --glossary glossary.json \
  --source 0 2>&1 | tee robustness_test.log
```

### Metrics Monitored

1. **FPS stability**: Consistent 20-30 FPS (no degradation)
2. **Memory**: <500 MB steady state
3. **Dropped frames**: <1% over 600s
4. **State machine**: No stuck states
5. **Prediction latency**: Stable (<50ms variance)

### Success Criteria

- ✓ FPS stable throughout
- ✓ No crashes or exceptions
- ✓ Memory doesn't increase >100 MB
- ✓ All recognized signs output correctly
- ✓ Sentence building works consistently

## 3.8 Naive User Evaluation

### Test Protocol

A user unfamiliar with the system attempts to sign 5 words from the glossary:

**Target**: 4/5 correct (80% recognition)

### Setup

1. User receives ~2 minute tutorial on:
   - Raise hand to start signing
   - Perform sign clearly
   - Lower hand to commit
   - System displays recognized word

2. User signs these words in sequence:
   - "hello"
   - "water"
   - "help"
   - "thankyou"
   - "goodbye"

3. System output recorded

### Evaluation Criteria

- **Success**: ≥4/5 correct recognitions
- **Failure**: <4/5 correct
- **Optional metrics**:
  - Time to first correct recognition
  - False positive rate during idle periods
  - User confidence/satisfaction

## 3.9 Checkpoint - Demo Ready

### Verification Checklist

- ✅ State machine test: Idle → signing → commit transitions work
- ✅ Bengali rendering: Conjunct characters display correctly
- ✅ Video I/O: Webcam and file input working
- ✅ Demo video: 60-90 second recording captured
- ✅ Robustness: 10-minute test passes
- ✅ User evaluation: Naive user achieves ≥80% accuracy
- ✅ FPS: Stable 20-30 FPS on CPU

### Demo-Ready Artifacts

- `demo.mp4`: Recorded demonstration (60-90 seconds)
- `glossary.json`: All 21 classes with Bengali labels
- `models/ishara_mlp.joblib`: Best-performing model
- `fonts/NotoSansBengali.ttf`: Bengali font (optional)

---

**Phase 3 Status**: ✓ Complete
**Next Phase**: International benchmarking (Phase 4)
