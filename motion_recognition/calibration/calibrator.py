import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, List

from motion_recognition.config import config
from motion_recognition.data.dataset import MotionDataset

class Calibrator:
    """Personalized calibration: fine‑tune model with user samples."""
    def __init__(self, base_model_path: Optional[Path] = None):
        self.config = config
        self.base_model_path = base_model_path or config.get_big_model_path()
        self.model = None
        self.small_model = None
        self.collected_samples: Dict[str, List[np.ndarray]] = {}
        self.dataset = MotionDataset(config)
        self.window_len = config.get_big_window_length_samples()
        self.window_step = config.get_big_window_step_samples()
        self.num_classes = len(config.ACTION_CLASSES)

    def _ensure_models_loaded(self):
        if self.model is None:
            if not self.base_model_path.exists():
                raise FileNotFoundError(f"Base model missing: {self.base_model_path}")
            self.model = tf.keras.models.load_model(self.base_model_path)
            print(f"Loaded base model: {self.base_model_path}")
        if self.small_model is None:
            from motion_recognition.models.cnn_1d import SmallWindowDetector
            det = SmallWindowDetector(self.config)
            try:
                det.load_model(self.config.get_small_model_path())
                self.small_model = det.model
                print("Loaded small detector")
            except Exception as e:
                print(f"Small detector load failed: {e}")
                self.small_model = None

    def collect_sample(self, action_label: str, sensor_data: np.ndarray, mode: int = 1, sample_idx: int = None) -> bool:
        self._ensure_models_loaded()
        if mode == 2:
            return self._collect_no_check(action_label, sensor_data)
        else:
            return self._collect_existence_only(action_label, sensor_data)

    def _collect_no_check(self, label: str, data: np.ndarray) -> bool:
        windows = self._extract_windows(data)
        if len(windows) == 0:
            print("Insufficient data")
            return False
        if label not in self.collected_samples:
            self.collected_samples[label] = []
        self.collected_samples[label].extend(windows.tolist())
        print(f"Collected '{label}' (no check), total windows: {len(self.collected_samples[label])}")
        return True

    def _collect_existence_only(self, label: str, data: np.ndarray) -> bool:
        windows = self._extract_windows(data)
        if len(windows) == 0:
            print("Insufficient data")
            return False
        if self.small_model is None:
            print("Small detector not available, falling back to no‑check")
            return self._collect_no_check(label, data)

        small_len = self.config.get_small_window_length_samples()
        good = []
        for win in windows:
            small = win[:small_len] if len(win) >= small_len else win
            inp = np.expand_dims(small, axis=0)
            pred = self.small_model.predict(inp, verbose=0)[0]
            if pred[1] > self.config.ACTION_EXISTENCE_THRESHOLD:
                good.append(win)
        if len(good) < len(windows) * 0.5:
            print(f"Not enough action windows: {len(good)}/{len(windows)}")
            return False
        if label not in self.collected_samples:
            self.collected_samples[label] = []
        self.collected_samples[label].extend(good)
        print(f"Collected '{label}' (existence check), kept {len(good)}/{len(windows)}, total {len(self.collected_samples[label])}")
        return True

    def _extract_windows(self, sensor_data: np.ndarray) -> np.ndarray:
        if sensor_data.shape[1] == 4:
            acc = sensor_data[:, 1:4]
        else:
            acc = sensor_data[:, :3]
        n = len(acc)
        if n < self.window_len:
            return np.array([])
        windows = []
        for start in range(0, n - self.window_len + 1, self.window_step):
            windows.append(acc[start:start+self.window_len])
        return np.array(windows)

    def run_calibration(self, epochs: int = 5, learning_rate: float = 0.001):
        self._ensure_models_loaded()
        if not self.collected_samples:
            print("No samples collected")
            return

        X, y = [], []
        for label, wins in self.collected_samples.items():
            idx = self.config.get_action_index(label)
            for w in wins:
                X.append(w)
                y.append(idx)
        X = np.array(X)
        y = np.array(y)
        print(f"Training data: {len(X)} windows from {len(self.collected_samples)} classes")

        for layer in self.model.layers:
            layer.trainable = False
        if len(self.model.layers) >= 2:
            self.model.layers[-1].trainable = True
            self.model.layers[-2].trainable = True
            print("Unfroze last 2 layers")
        else:
            for layer in self.model.layers:
                layer.trainable = True
            print("Unfroze all layers")

        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Fine‑tuning: lr={learning_rate}, epochs={epochs}")
        self.model.fit(X, y, epochs=epochs, batch_size=min(32, len(X)), verbose=1)
        print("Fine‑tuning finished")

    def save_calibrated_model(self, user_id: str = "default") -> Path:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        model_dir = self.config.MODEL_SAVE_DIR / "calibrated"
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{user_id}_big_classifier.h5"
        self.model.save(path)
        print(f"Saved calibrated model to {path}")
        return path

    def reset(self):
        self.collected_samples.clear()
        self.model = None
        self.small_model = None
        print("Calibrator reset")