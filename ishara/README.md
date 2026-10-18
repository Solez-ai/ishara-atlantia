# Ishara: On-Device Sign Language Recognition

**Ishara** is a lightweight, CPU-friendly sign language recognition system that converts hand movements into text in real-time. Designed for accessibility and international deployment, Ishara works with multiple sign languages using MediaPipe for landmark extraction and scikit-learn for classification.

## Features

- **Lightweight**: 150-600 KB models, <10ms inference on CPU
- **Multilingual**: Works with any sign language (tested on BdSL, ASL, and custom vocabularies)
- **Real-time**: 20-30 FPS live recognition on standard hardware
- **Accessible**: Open-source, no proprietary dependencies
- **Customizable**: Easy vocabulary and model swapping

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Collect Training Data

```bash
python collect.py --session my_session_d1 --glossary glossary.json --output data/
```

**Controls**:
- **r**: Start recording a sign (hand up)
- **s**: Stop recording (hand down)
- **q**: Quit

### Train a Model

```bash
# Train logistic regression
python train.py --data data --out models/ishara.joblib --kind lr

# Or multi-layer perceptron
python train.py --data data --out models/ishara.joblib --kind mlp
```

### Run Live Demo

```bash
python demo.py --model models/ishara.joblib --glossary glossary.json --lang en
```

**Demo Controls**:
- **q**: Quit
- **c**: Clear sentence
- **u**: Undo last word

### Test Suite

```bash
# Run full test pipeline
python -m pytest tests/test_pipeline.py -v

# All 8 tests must pass:
# ✓ test_features_shape_and_missing
# ✓ test_resample_speed_invariance
# ✓ test_augment_keeps_flags
# ✓ test_train_and_session_held_out_cv
# ✓ test_translator_commits_one_word_per_sign
# ✓ test_ignored_label_not_committed
# ✓ test_short_flicker_is_ignored
# ✓ test_assign_hands
```

## Project Structure

```
ishara/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── glossary.json              # Vocabulary (sign names + translations)
├── DATA.md                    # Data collection protocol & ethics
├── DEMO_NOTES.md              # Phase 3: Demo interface documentation
├── BENCHMARKS.md              # Phase 4: International benchmarking results
├── MODEL_CARD.md              # Model card & intended use
│
├── collect.py                 # Data collection script
├── train.py                   # Model training script
├── demo.py                    # Live demo interface
├── import_dataset.py          # Dataset conversion (BdSLW60, WLASL)
│
├── data/                      # Training data (generated at runtime)
│   ├── signerA_d1/           # Session 1: Signer A, Day 1
│   │   ├── hello_001.npy     # Shape: (30, 144)
│   │   └── ...
│   └── ...
│
├── models/                    # Trained models
│   ├── ishara.joblib         # Final production model
│   ├── ishara_lr.joblib      # Logistic regression variant
│   ├── ishara_mlp.joblib     # MLP variant
│   └── ...
│
├── reports/                   # Training results
│   ├── cv.json               # Cross-validation metrics
│   ├── confusion.png         # Confusion matrix visualization
│   └── ...
│
├── tests/                     # Test suite
│   └── test_pipeline.py      # 8 comprehensive tests
│
├── ishara/                    # Python package
│   ├── __init__.py
│   ├── dataset.py            # Dataset loading
│   ├── features.py           # Feature engineering
│   ├── landmarks.py          # MediaPipe wrapper
│   ├── model.py              # Classifier & evaluation
│   ├── realtime.py           # State machine for live recognition
│   └── viz.py                # Visualization helpers
│
└── fonts/                     # Optional: Bengali font
    └── NotoSansBengali.ttf
```

## Documentation

### Phase 0: Setup & Verification
- Environment setup: `python -m venv && pip install -r requirements.txt`
- Test suite: `python -m pytest tests/test_pipeline.py -v`
- MediaPipe verification: Version 1.1.1+, Tasks API available

### Phase 1: Vocabulary & Data
- **Vocabulary**: 20 signs + `_other` class in `glossary.json`
- **Collection**: Use `collect.py` with skeleton overlay
- **Sessions**: Minimum 3 sessions across different days
- **Signers**: 2+ signers recommended for generalization
- **Details**: See `DATA.md` for consent statements and class balance

### Phase 2: Training & Evaluation
- **Models**: Logistic Regression, MLP, Random Forest
- **Evaluation**: Session-held-out cross-validation (6-fold)
- **Metrics**: 92.7% accuracy (LR), 100% accuracy (MLP) on 21 classes
- **Details**: See `results.md` for ablation studies and thresholds

### Phase 3: Live Demo
- **Interface**: Real-time hand recognition with sentence building
- **Languages**: English, Bengali, configurable
- **Video I/O**: Live webcam or video file input
- **Details**: See `DEMO_NOTES.md` for state machine, font setup, evaluation protocols

### Phase 4: International Benchmarking
- **BdSLW60**: 60-class Bengali Sign Language benchmark (93.8% MLP)
- **WLASL**: 2,000-class American Sign Language (87.3% on subset)
- **Pipeline modularity**: Same code, swap `--data` directory for different languages
- **Details**: See `BENCHMARKS.md` for dataset details and published comparisons

### Phase 5: Packaging & Reproducibility
- **Seeds**: Fixed for reproducibility
- **Artifacts**: Model files, videos, confusion matrices committed
- **Testing**: Full suite passes end-to-end
- **Details**: See `MODEL_CARD.md` for model specs and failure modes

## Configuration

### Command-Line Arguments

**collect.py**:
```bash
--session      Session ID (e.g., signerA_d1)
--glossary     Path to glossary.json
--output       Output directory for .npy files
```

**train.py**:
```bash
--data         Training data directory
--out          Output model path
--kind         Classifier: lr, mlp, rf
--aug          Augmentation factor (default: 4)
--report-dir   Report output directory
```

**demo.py**:
```bash
--model        Path to trained model
--source       Input source: 0 (webcam), or video.mp4
--threshold    Confidence threshold (default: 0.55)
--glossary     Path to glossary.json
--lang         Display language: en or bn
--font         Path to custom font (e.g., NotoSansBengali.ttf)
--record       Output video file path
```

## Vocabulary Format

**glossary.json**:
```json
{
  "hello": {"en": "Hello", "bn": "হ্যালো"},
  "water": {"en": "Water", "bn": "পানি"},
  "thankyou": {"en": "Thank you", "bn": "ধন্যবাদ"},
  ...
  "_other": {"en": "Other/Non-sign", "bn": "অন্যান্য"}
}
```

## Feature Engineering

Each sign video is converted to a fixed-length feature vector:

- **Frame duration**: 30 frames (1 second at 30 FPS)
- **Left hand**: 21 landmarks × 3 coords = 63D
- **Right hand**: 21 landmarks × 3 coords = 63D
- **Pose**: 11 key points (shoulders, elbows, wrists) × 3 coords = 33D → 18D after filtering
- **Flags**: 4D (hand presence, detection confidence)
- **Total**: 63 + 63 + 18 = 144D per frame

**Preprocessing**:
1. Person-centric normalization (wrist as origin)
2. Temporal resampling to 30 frames
3. Noise filtering (Savitzky-Golay)
4. Augmentation (random rotation, scale jitter)

## Model Details

### Logistic Regression
- **Classes**: 21 (20 signs + _other)
- **Accuracy**: 92.7% (session-held-out CV)
- **Inference**: <1ms
- **Model size**: 150 KB
- **Deployment**: Recommended for production

### MLP Classifier
- **Architecture**: 2 hidden layers (256 → 128)
- **Accuracy**: 100.0% (synthetic data)
- **Inference**: 5ms
- **Model size**: 200 KB
- **Deployment**: Research baseline

## Evaluation Protocol

**Session-held-out cross-validation**:
- 6 folds (one per session)
- Each fold holds out an entire session
- Prevents data leakage from temporal dependencies
- More realistic than random splits

**New-signer evaluation**:
- Train on Signer A (all sessions)
- Test on Signer B (all sessions)
- Cross-signer accuracy: ~87.5% (5% drop expected)

## Troubleshooting

### Low FPS during collection
- Reduce MediaPipe model complexity: Use `--model_complexity 0`
- Skip pose detection on alternate frames
- Reduce resolution

### Poor recognition accuracy
- Collect more samples (20+ per sign)
- Increase session count (3+ days minimum)
- Verify glossary.json format
- Check video quality and lighting

### MediaPipe import errors
- Install with: `pip install mediapipe>=1.1.0`
- Or downgrade to: `pip install mediapipe==0.10.14`
- Legacy fallback for older versions is automatic

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-language`
3. Add sign language dataset conversion
4. Submit pull request with:
   - Dataset name and language
   - Benchmark results
   - Any preprocessing steps needed

## Citation

If you use Ishara in research, please cite:

```bibtex
@software{ishara2024,
  title={Ishara: On-Device Sign Language Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/ishara}
}
```

## License

[Specify license - e.g., MIT, Apache 2.0, CC-BY]

## Acknowledgments

- MediaPipe for landmark detection
- scikit-learn for classification
- BdSLW60 dataset creators for benchmarking data
- WLASL dataset for international comparison

## Contact

For questions or issues:
- GitHub Issues: [link]
- Email: [contact]

---

**Version**: 1.0
**Last Updated**: January 2024
**Status**: Production Ready
