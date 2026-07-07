import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
import json
from pathlib import Path

class MotionTrainer:
    def __init__(self, config, model_builder, model_type='big'):
        self.config = config
        self.model_builder = model_builder
        self.model_type = model_type
        self.model = None
        self.history = None
        self.model_save_dir = config.MODEL_SAVE_DIR
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self, windows: np.ndarray, labels: np.ndarray):
        if len(windows) == 0:
            raise ValueError("Empty data")
        idx = np.arange(len(windows))
        np.random.shuffle(idx)
        return windows[idx], labels[idx]

    def train(self, train_windows, train_labels, val_windows=None, val_labels=None,
              epochs=None, batch_size=None) -> Dict[str, Any]:
        if epochs is None:
            epochs = self.config.EPOCHS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        print(f"Training {self.model_type} model: {epochs} epochs, batch size {batch_size}")
        print(f"Training samples: {len(train_windows)}")
        if val_windows is not None:
            print(f"Validation samples: {len(val_windows)}")

        train_windows, train_labels = self.prepare_training_data(train_windows, train_labels)
        print("Building model...")
        self.model = self.model_builder.compile_model()
        callbacks = self._create_callbacks()
        val_data = (val_windows, val_labels) if val_windows is not None else None

        start = time.time()
        history = self.model.fit(
            train_windows, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        print(f"Training finished in {time.time()-start:.1f}s")
        self.history = history.history
        self._print_summary()
        return self.history

    def _create_callbacks(self) -> list:
        callbacks = []
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=self.config.PATIENCE, min_delta=self.config.MIN_DELTA,
            restore_best_weights=True, verbose=1))
        if self.model_type == 'small':
            model_path = self.config.get_small_model_path()
        else:
            model_path = self.config.get_big_model_path()
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor='loss', save_best_only=True, mode='min', verbose=1))
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1))
        return callbacks

    def _print_summary(self):
        if self.history is None:
            return
        print("\n" + "="*60)
        print("Training summary")
        print("="*60)
        print(f"Epochs: {len(self.history['loss'])}")
        print(f"Final loss: {self.history['loss'][-1]:.4f}")
        print(f"Final accuracy: {self.history['accuracy'][-1]:.4f}")
        if 'val_loss' in self.history:
            print(f"Val loss: {self.history['val_loss'][-1]:.4f}")
            print(f"Val accuracy: {self.history['val_accuracy'][-1]:.4f}")

    def evaluate(self, test_windows: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        if self.model is None:
            print("Model not loaded, attempting to load...")
            if self.model_type == 'small':
                self.model_builder.load_model()
            else:
                self.model_builder.load_model()
            self.model = self.model_builder.model
        if self.model is None:
            raise ValueError("Cannot load model")
        print(f"Evaluating {self.model_type} model on {len(test_windows)} samples")
        results = self.model.evaluate(test_windows, test_labels, verbose=0)
        if isinstance(results, list):
            metrics = dict(zip(self.model.metrics_names, results))
        else:
            metrics = {'loss': results}
        print("\nEvaluation results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        return metrics

    def save_training_history(self, filename=None):
        if self.history is None:
            return
        if filename is None:
            filename = f"{self.model_type}_history"
        save_path = self.config.LOGS_DIR / f"{filename}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: [float(vv) for vv in v] for k, v in self.history.items()}
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Training history saved to {save_path}")