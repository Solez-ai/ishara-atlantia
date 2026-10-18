# Ishara Sign Language Dataset

## Consent and Ethics

All signers have provided explicit written consent for their sign language samples to be used for training and evaluation of the Ishara sign language recognition system.

**Consent Statement (Template):**
> "I consent to my sign language samples being used for training and evaluation of the Ishara sign language recognition system. I understand that these samples may be used to develop assistive technology for Deaf communities and to support research in sign language recognition."

## Data Collection Sessions

### Session: signerA_d1
- **Date**: 2024-01-15
- **Signer**: Signer A
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Controlled lighting, professional camera setup

### Session: signerA_d2
- **Date**: 2024-01-17
- **Signer**: Signer A
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Different time of day, varied lighting

### Session: signerA_d3
- **Date**: 2024-01-20
- **Signer**: Signer A
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Casual environment, natural lighting

### Session: signerB_d1
- **Date**: 2024-01-22
- **Signer**: Signer B
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Different signer for generalization testing

### Session: signerB_d2
- **Date**: 2024-01-24
- **Signer**: Signer B
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Varied camera angle

### Session: signerB_d3
- **Date**: 2024-01-26
- **Signer**: Signer B
- **Duration**: ~45 minutes
- **Total Samples**: 420 (20 signs × 21 samples)
- **Signs Collected**: All 20 signs + _other class
- **Notes**: Different background

## Class Distribution

| Sign | Session 1 | Session 2 | Session 3 | Session 4 | Session 5 | Session 6 | Total |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| hello | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| goodbye | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| thankyou | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| sorry | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| yes | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| no | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| please | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| help | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| water | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| food | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| name | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| doctor | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| hospital | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| school | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| home | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| work | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| family | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| friend | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| happy | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| tired | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| _other | 21 | 21 | 21 | 21 | 21 | 21 | 126 |
| **TOTAL** | **441** | **441** | **441** | **441** | **441** | **441** | **2646** |

## Class Balance Analysis

- **Minimum samples per class**: 126
- **Maximum samples per class**: 126
- **Ratio (max/min)**: 1.0 (perfectly balanced)
- **Status**: ✓ All classes within 2x threshold

## Data Format

All samples are stored as NumPy arrays with shape (30, 144):
- **30**: Number of video frames per sign episode
- **144**: Feature dimension (63 dims per hand × 2 + 18 pose dims)

**File structure:**
```
data/
├── signerA_d1/
│   ├── hello_001.npy
│   ├── hello_002.npy
│   └── ...
├── signerA_d2/
│   └── ...
└── ...
```

## Notes for Reproduction

- All signs were collected at 30 FPS
- Skeleton landmarks normalized to person-centric coordinates
- Pose and hand detection used MediaPipe
- Each sample represents a single signed word with clear hand raise/lower transitions
