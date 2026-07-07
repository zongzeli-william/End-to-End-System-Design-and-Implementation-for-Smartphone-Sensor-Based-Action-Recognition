import numpy as np
import time
from collections import deque
from pathlib import Path
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RealtimePredictor:
    """Two‑stage real‑time predictor for 10Hz data."""
    def __init__(self, config, small_model_path=None, big_model_path=None):
        print("Initializing real‑time predictor...")
        self.config = config
        from motion_recognition.models.cnn_1d import SmallWindowDetector, BigWindowClassifier

        self.small_detector = SmallWindowDetector(config)
        if small_model_path is None:
            small_model_path = config.get_small_model_path()
        try:
            self.small_detector.load_model(small_model_path)
            self.small_model = self.small_detector.model
            print("✓ Small detector loaded")
        except Exception as e:
            print(f"✗ Small detector failed: {e}")
            self.small_model = None

        self.big_classifier = BigWindowClassifier(config)
        if big_model_path is None:
            big_model_path = config.get_big_model_path()
        try:
            self.big_classifier.load_model(big_model_path)
            self.big_model = self.big_classifier.model
            print("✓ Big classifier loaded")
        except Exception as e:
            print(f"✗ Big classifier failed: {e}")
            self.big_model = None

        self.small_window_len = config.get_small_window_length_samples()
        self.big_window_len = config.get_big_window_length_samples()
        self.buffer = np.zeros((0, config.NUM_FEATURES))
        self.action_present = False
        self.action_start_time = None
        self.current_action = 'no_action'
        self.action_confidence = 0.0
        self.big_pred_history = deque(maxlen=5)
        self.last_output_time = 0
        print("Predictor ready")

    def process_incremental(self, new_data):
        if new_data.ndim == 1:
            new_data = new_data.reshape(1, -1)
        if new_data.shape[1] == 4:
            accel = new_data[:, 1:4]
        else:
            accel = new_data[:, :3]
        self.buffer = np.vstack([self.buffer, accel])
        max_len = self.big_window_len * 3
        if len(self.buffer) > max_len:
            self.buffer = self.buffer[-max_len:]

        result = {
            'timestamp': time.time(),
            'action': 'no_action',
            'confidence': 0.0,
            'action_present': False,
            'top1_action': 'no_action',
            'top1_confidence': 0.0,
            'top2_action': 'no_action',
            'top2_confidence': 0.0,
        }

        if len(self.buffer) >= self.small_window_len:
            small_win = self.buffer[-self.small_window_len:]
            if self.small_model is not None:
                inp = np.expand_dims(small_win, axis=0)
                pred = self.small_model.predict(inp, verbose=0)[0]
                action_prob = pred[1]
            else:
                action_prob = self._energy_detection(small_win)

            was = self.action_present
            self.action_present = action_prob > self.config.ACTION_EXISTENCE_THRESHOLD
            now = time.time()
            if self.action_present and not was:
                self.action_start_time = now
            elif not self.action_present:
                self.action_start_time = None

            if len(self.buffer) >= self.big_window_len:
                big_win = self.buffer[-self.big_window_len:]
                if self.big_model is not None:
                    inp = np.expand_dims(big_win, axis=0)
                    pred = self.big_model.predict(inp, verbose=0)[0]
                    self.big_pred_history.append(pred)

                if (self.action_start_time is not None and
                    now - self.action_start_time >= self.config.MIN_ACTION_DURATION):
                    if self.big_pred_history:
                        avg = np.mean(self.big_pred_history, axis=0)
                        top_idx = np.argsort(avg)[-2:][::-1]
                        t1, t2 = top_idx[0], top_idx[1]
                        a1 = self.config.get_action_label(t1)
                        c1 = float(avg[t1])
                        a2 = self.config.get_action_label(t2)
                        c2 = float(avg[t2])
                        if a1 == 'no_action':
                            final_a = a2
                            final_c = c2
                        else:
                            final_a = a1
                            final_c = c1
                        result['action'] = final_a
                        result['confidence'] = final_c
                        result['action_present'] = True
                        result['top1_action'] = a1
                        result['top1_confidence'] = c1
                        result['top2_action'] = a2
                        result['top2_confidence'] = c2
        return result

    def _energy_detection(self, window):
        mag = np.sqrt(np.sum(window**2, axis=1))
        return min(1.0, np.var(mag) / 2.0)

    def reset(self):
        self.buffer = np.zeros((0, self.config.NUM_FEATURES))
        self.action_present = False
        self.action_start_time = None
        self.current_action = 'no_action'
        self.action_confidence = 0.0
        self.big_pred_history.clear()
        print("Predictor reset")

# ==================== Module‑level functions (for MATLAB) ====================
_predictor_instance = None
_calibrator_instance = None
_original_predictor = None
_custom_predictor = None

def init_predictor(config=None, small_path=None, big_path=None):
    global _predictor_instance
    if _predictor_instance is None:
        if config is None:
            from motion_recognition.config import config
        _predictor_instance = RealtimePredictor(config, small_path, big_path)
    return _predictor_instance

def predict(sensor_data):
    try:
        p = init_predictor()
        r = p.process_incremental(sensor_data)
        return {'action': r['action'], 'confidence': r['confidence'], 'action_present': r['action_present']}
    except Exception as e:
        return {'error': str(e)}

def reset():
    global _predictor_instance
    if _predictor_instance:
        _predictor_instance.reset()
    return {'status': 'reset'}

def test_connection(test_data=None):
    try:
        p = init_predictor()
        if test_data is None:
            test_data = np.random.randn(5,3).astype(np.float32)
        r = p.process_incremental(test_data)
        return {'status': 'success', 'result': r}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

# Calibration
from motion_recognition.calibration.calibrator import Calibrator
def get_calibrator():
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = Calibrator()
    return _calibrator_instance

def collect_calibration_sample(action_label, sensor_data, mode=1, sample_idx=None):
    try:
        c = get_calibrator()
        ok = c.collect_sample(action_label, sensor_data, mode, sample_idx)
        return {'success': ok, 'message': 'Success' if ok else 'Failed'}
    except Exception as e:
        return {'success': False, 'message': str(e)}

def reset_calibration():
    try:
        c = get_calibrator()
        c.reset()
        return {'status': 'success', 'message': 'Calibrator reset'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def finish_calibration(user_id='default', epochs=5):
    try:
        c = get_calibrator()
        c.run_calibration(epochs=epochs)
        path = c.save_calibrated_model(user_id)
        global _calibrator_instance
        _calibrator_instance = None
        return {'status': 'success', 'model_path': str(path)}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Comparison
def _get_original_predictor():
    global _original_predictor
    if _original_predictor is None:
        from motion_recognition.config import config
        _original_predictor = RealtimePredictor(config)
        _original_predictor.big_model = tf.keras.models.load_model(config.get_big_model_path())
        print(f"Loaded original model from {config.get_big_model_path()}")
    return _original_predictor

def _get_custom_predictor(custom_path=None):
    global _custom_predictor
    if _custom_predictor is None:
        from motion_recognition.config import config
        if custom_path is None:
            custom_path = config.MODEL_SAVE_DIR / "calibrated" / "current_user_big_classifier.h5"
        if not custom_path.exists():
            return None
        _custom_predictor = RealtimePredictor(config)
        _custom_predictor.big_model = tf.keras.models.load_model(custom_path)
        print(f"Loaded custom model from {custom_path}")
    return _custom_predictor

def predict_original(sensor_data):
    try:
        p = _get_original_predictor()
        r = p.process_incremental(sensor_data)
        return {
            'action': r['action'], 'confidence': r['confidence'], 'action_present': r['action_present'],
            'top1_action': r.get('top1_action', r['action']),
            'top1_confidence': r.get('top1_confidence', r['confidence']),
            'top2_action': r.get('top2_action', 'no_action'),
            'top2_confidence': r.get('top2_confidence', 0.0)
        }
    except Exception as e:
        return {'error': str(e)}

def predict_custom(sensor_data, custom_path=None):
    try:
        p = _get_custom_predictor(custom_path)
        if p is None:
            return {'error': 'Custom model not found'}
        r = p.process_incremental(sensor_data)
        return {
            'action': r['action'], 'confidence': r['confidence'], 'action_present': r['action_present'],
            'top1_action': r.get('top1_action', r['action']),
            'top1_confidence': r.get('top1_confidence', r['confidence']),
            'top2_action': r.get('top2_action', 'no_action'),
            'top2_confidence': r.get('top2_confidence', 0.0)
        }
    except Exception as e:
        return {'error': str(e)}