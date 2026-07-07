import os
from pathlib import Path

class Config:
    """Configuration for 10Hz sampling and two-stage detection."""

    # ========== Paths ==========
    PROJECT_ROOT = Path(__file__).parent
    MOTION_DATA_ROOT = PROJECT_ROOT.parent / "motion_data"

    TRAIN_DATA_BASE = MOTION_DATA_ROOT / "training_data"
    TRAIN_DATA_PATHS = {d.name: d for d in TRAIN_DATA_BASE.iterdir() if d.is_dir()}

    VAL_DATA_PATH = MOTION_DATA_ROOT / "validation_set"

    OUTPUT_ROOT = PROJECT_ROOT / "output"
    MODEL_SAVE_DIR = OUTPUT_ROOT / "models"
    PROCESSED_DATA_DIR = OUTPUT_ROOT / "processed_data"
    LOGS_DIR = OUTPUT_ROOT / "logs"

    # ========== Data format ==========
    MOTION_COLUMNS = ['timestamp', 'accel_x', 'accel_y', 'accel_z']
    ANNOTATION_COLUMNS = ['start_time', 'end_time', 'action_label', 'confidence']

    SAMPLING_RATE = 10          # Hz
    NUM_FEATURES = 3

    # Two‑stage window parameters
    SMALL_WINDOW_SIZE = 0.3         # seconds (3 samples)
    SMALL_OVERLAP_RATIO = 0.67
    BIG_WINDOW_SIZE = 1.2           # seconds (12 samples)
    BIG_OVERLAP_RATIO = 0.5

    MIN_WINDOW_SIZE = 0.2
    MAX_WINDOW_SIZE = 3.0

    ACTION_CLASSES = [
        'right_down',
        'left_down',
        'circle_clock',
        'circle_anticlock',
        'no_action'
    ]

    # Detection thresholds
    ACTION_EXISTENCE_THRESHOLD = 0.4
    ACTION_CLASS_CONFIDENCE = 0.5
    MIN_ACTION_DURATION = 0.4       # seconds
    DEBOUNCE_TIME = 0.5

    # Training
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2

    MODEL_NAME_SMALL = "small_detector"
    MODEL_NAME_BIG = "big_classifier"
    MODEL_EXT = ".h5"

    PATIENCE = 10
    MIN_DELTA = 0.001

    # Inference
    CONFIDENCE_THRESHOLD = 0.7
    ACTION_MIN_DURATION = 0.3
    DEBOUNCE_TIME = 0.5

    RANDOM_SEED = 42
    USE_GPU = True

    def __init__(self):
        self._create_directories()

    def _create_directories(self):
        for d in [self.MODEL_SAVE_DIR, self.PROCESSED_DATA_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def get_small_window_length_samples(self):
        return int(self.SMALL_WINDOW_SIZE * self.SAMPLING_RATE)

    def get_small_window_step_samples(self):
        wlen = self.get_small_window_length_samples()
        return max(1, int(wlen * (1 - self.SMALL_OVERLAP_RATIO)))

    def get_big_window_length_samples(self):
        return int(self.BIG_WINDOW_SIZE * self.SAMPLING_RATE)

    def get_big_window_step_samples(self):
        wlen = self.get_big_window_length_samples()
        return max(1, int(wlen * (1 - self.BIG_OVERLAP_RATIO)))

    def get_action_index(self, label):
        return self.ACTION_CLASSES.index(label)

    def get_action_label(self, idx):
        if 0 <= idx < len(self.ACTION_CLASSES):
            return self.ACTION_CLASSES[idx]
        return 'unknown'

    def get_train_data_dirs(self):
        return list(self.TRAIN_DATA_PATHS.values())

    def get_small_model_path(self):
        return self.MODEL_SAVE_DIR / f"{self.MODEL_NAME_SMALL}{self.MODEL_EXT}"

    def get_big_model_path(self):
        return self.MODEL_SAVE_DIR / f"{self.MODEL_NAME_BIG}{self.MODEL_EXT}"

config = Config()