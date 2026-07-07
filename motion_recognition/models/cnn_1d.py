import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np

class SmallWindowDetector:
    """Binary classifier for action presence (input shape (3,3))."""
    def __init__(self, config, model_name="small_detector"):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.window_length = config.get_small_window_length_samples()
        self.num_features = config.NUM_FEATURES
        self.num_classes = 2
        print(f"Small detector config: input=({self.window_length},{self.num_features})")

    def build_model(self):
        self.model = models.Sequential([
            layers.Input(shape=(self.window_length, self.num_features)),
            layers.Flatten(),
            layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(8, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return self.model

    def compile_model(self, lr=0.001):
        if self.model is None:
            self.build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Small detector compiled")
        return self.model

    def summary(self):
        if self.model:
            self.model.summary()

    def save_model(self, path=None):
        if self.model is None:
            return
        if path is None:
            path = self.config.get_small_model_path()
        self.model.save(path)
        print(f"Saved small detector to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.config.get_small_model_path()
        try:
            self.model = tf.keras.models.load_model(path)
            print(f"Loaded small detector from {path}")
        except Exception as e:
            print(f"Failed to load small detector: {e}, building new")
            self.compile_model()

class BigWindowClassifier:
    """Multi‑class classifier (input shape (12,3))."""
    def __init__(self, config, model_name="big_classifier"):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.window_length = config.get_big_window_length_samples()
        self.num_features = config.NUM_FEATURES
        self.num_classes = len(config.ACTION_CLASSES)
        print(f"Big classifier config: input=({self.window_length},{self.num_features}), classes={self.num_classes}")

    def build_model(self):
        inputs = layers.Input(shape=(self.window_length, self.num_features))

        x = layers.Conv1D(32, kernel_size=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(64, kernel_size=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(128, kernel_size=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        self.model = models.Model(inputs, outputs)
        return self.model

    def compile_model(self, lr=0.001):
        if self.model is None:
            self.build_model()
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Big classifier compiled")
        return self.model

    def summary(self):
        if self.model:
            self.model.summary()

    def save_model(self, path=None):
        if self.model is None:
            return
        if path is None:
            path = self.config.get_big_model_path()
        self.model.save(path)
        print(f"Saved big classifier to {path}")

    def load_model(self, path=None):
        if path is None:
            path = self.config.get_big_model_path()
        try:
            self.model = tf.keras.models.load_model(path)
            print(f"Loaded big classifier from {path}")
        except Exception as e:
            print(f"Failed to load big classifier: {e}, building new")
            self.compile_model()

def create_small_detector(config):
    return SmallWindowDetector(config)

def create_big_classifier(config):
    return BigWindowClassifier(config)