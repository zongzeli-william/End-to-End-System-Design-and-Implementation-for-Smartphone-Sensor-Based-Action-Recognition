# Mobile Sensor Motion Recognition System

A real-time gesture recognition system based on MATLAB frontend + Python backend. Supports training, evaluation, real-time demo, and personalised calibration.

## Requirements

- Python 3.8+, TensorFlow 2.x, NumPy, Pandas
- MATLAB R2020b+ with MATLAB Mobile plugin
- Virtual environment (conda or venv) recommended

## Quick Start

1. Prepare Data

Place collected data in the following structure:
motion_data/
├── training_data/ # Training data (all subfolders auto‑scanned)
│ ├── single_motion/
│ │ ├── motion/ # Sensor CSV files, e.g., m01.csv
│ │ └── annotation/ # Label CSV files, e.g., a01.csv
│ ├── mult_motion/
│ └── ... (any folder name)
└── validation_set/ # Optional validation set
├── motion/
└── annotation/


- Sensor file columns: `timestamp, accel_x, accel_y, accel_z`
- Annotation file columns: `start_time, end_time, action_label, confidence`

2. Install Dependencies

```bash
pip install tensorflow numpy pandas

3. Train Models

# Default: train both small and big detectors (using all data)
python motion_recognition/scripts/main.py --mode train

# Train only small or big detector
python main.py --mode train --model_type small
python main.py --mode train --model_type big

# Specify number of epochs
python main.py --mode train --epochs 50

# Use only specific data folders
python main.py --mode train --include single_motion mult_motion
# Exclude certain folders
python main.py --mode train --exclude mult_single_motion

4. Evaluate Models

# Evaluate both small and big (default)
python main.py --mode evaluate

# Evaluate only small or big
python main.py --mode evaluate --model_type small
python main.py --mode evaluate --model_type big

# Compare evaluation (output top‑3 per action)
python main.py --mode evaluate --compare
# Or use compare mode (train big first then evaluate)
python main.py --mode compare

5. Real‑time Demo (Playback Historical Data)

# Use the first file in validation set
python main.py --mode realtime

# Specify a CSV file
python main.py --mode realtime --file "E:\...\motion_data\validation_set\motion\m4.csv"

6. Full Pipeline (Train + Evaluate + Real‑time Demo)

python main.py --mode all


MATLAB Frontend Usage
Open MATLAB and run personalised.m.

Choose mode as prompted:

Mode 1: Collect actions (check existence, keep valid windows) – for personalised calibration.

Mode 2: Direct collection (no validation) – keep all windows.

Mode 3: Skip calibration and directly use existing model for real‑time comparison.

After calibration, the script automatically enters real‑time comparison mode, showing predictions from both original and personalised models.

Additional Notes
All outputs (models, logs) are saved under the output/ directory.

Personalised models are saved in output/models/calibrated/.

Model files are small_detector.h5 and big_classifier.h5.

To adjust sensitivity, modify threshold parameters in motion_recognition/config.py (e.g., ACTION_EXISTENCE_THRESHOLD, MIN_ACTION_DURATION).


IndividualProject/
├── motion_recognition/             # Python backend
│   ├── config.py                          # Configuration parameters
│   ├── data/dataset.py                # Data loading & window generation
│   ├── models/cnn_1d.py            # Small detector & big classifier
│   ├── training/trainer.py            # Training & evaluation
│   ├── inference/                        # Real‑time inference interface
│   ├── calibration/calibrator.py   # Personalised fine‑tuning
│   └── scripts/main.py                # Main command‑line entry
├── motion_data/                       # Data directory
│   ├── training_data/                 # Training subfolders
│   └── validation_set/                # Validation set
├── output/                                # Model & log outputs
└── MATLAB/                             # Frontend scripts
    ├── personalised.m                # Calibration & real‑time comparison
    └── autocollection.m             # Semi‑automatic data collection
