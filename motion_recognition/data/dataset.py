import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class MotionDataset:
    """Dataset loader for small and big windows."""

    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLING_RATE
        self.small_window_len = config.get_small_window_length_samples()
        self.small_window_step = config.get_small_window_step_samples()
        self.big_window_len = config.get_big_window_length_samples()
        self.big_window_step = config.get_big_window_step_samples()

    # ---------- Basic I/O ----------
    def load_sensor_data(self, motion_file: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(motion_file)
            col_map = {}
            for col in df.columns:
                low = col.lower()
                if 'timestamp' in low or 'time' == low:
                    col_map[col] = 'timestamp'
                elif 'accel_x' in low or 'ax' in low or low == 'x':
                    col_map[col] = 'accel_x'
                elif 'accel_y' in low or 'ay' in low or low == 'y':
                    col_map[col] = 'accel_y'
                elif 'accel_z' in low or 'az' in low or low == 'z':
                    col_map[col] = 'accel_z'
            if col_map:
                df = df.rename(columns=col_map)
            if 'timestamp' not in df.columns:
                df['timestamp'] = np.arange(len(df)) / self.sample_rate
            for col in ['accel_x', 'accel_y', 'accel_z']:
                if col not in df.columns:
                    df[col] = 0.0
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            for col in ['accel_x', 'accel_y', 'accel_z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna().sort_values('timestamp').reset_index(drop=True)
            return df[['timestamp', 'accel_x', 'accel_y', 'accel_z']]
        except Exception as e:
            print(f"Failed to load {motion_file.name}: {e}")
            return pd.DataFrame()

    def load_annotation_data(self, annotation_file: Path) -> pd.DataFrame:
        try:
            if not annotation_file.exists():
                return pd.DataFrame()
            df = pd.read_csv(annotation_file)
            col_map = {}
            for col in df.columns:
                low = col.lower()
                if 'start' in low:
                    col_map[col] = 'start_time'
                elif 'end' in low:
                    col_map[col] = 'end_time'
                elif 'label' in low or 'action' in low:
                    col_map[col] = 'action_label'
            if col_map:
                df = df.rename(columns=col_map)
            for col in ['start_time', 'end_time', 'action_label']:
                if col not in df.columns:
                    if col == 'action_label':
                        df['action_label'] = 'no_action'
                    else:
                        return pd.DataFrame()
            df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
            df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
            df = df.dropna()
            return df[['start_time', 'end_time', 'action_label']]
        except Exception as e:
            print(f"Failed to load annotation {annotation_file.name}: {e}")
            return pd.DataFrame()

    def find_annotation_file(self, motion_file: Path, data_dir: Path) -> Path:
        anno_dir = data_dir / "annotation"
        if not anno_dir.exists():
            return None
        stem = motion_file.stem
        if stem.startswith('m'):
            cand = anno_dir / f"a{stem[1:]}.csv"
        else:
            cand = anno_dir / f"{stem}.csv"
        if cand.exists():
            return cand
        for suffix in ['', '_annotation', '_label']:
            cand = anno_dir / f"{stem}{suffix}.csv"
            if cand.exists():
                return cand
        return None

    # ---------- Window generation ----------
    def _get_window_label(self, start: float, end: float, ann: pd.DataFrame) -> int:
        if ann.empty:
            return self.config.get_action_index('no_action')
        overlap = ann[(ann['end_time'] > start) & (ann['start_time'] < end)]
        if overlap.empty:
            return self.config.get_action_index('no_action')
        label = overlap['action_label'].value_counts().index[0]
        try:
            return self.config.get_action_index(label)
        except ValueError:
            return self.config.get_action_index('no_action')

    def _get_existence_label(self, start: float, end: float, ann: pd.DataFrame) -> int:
        if ann.empty:
            return 0
        overlap = ann[(ann['end_time'] > start) & (ann['start_time'] < end)]
        return 1 if not overlap.empty else 0

    def create_windows_from_file(self, motion_file: Path, data_dir: Path,
                                 window_type='big') -> Tuple[np.ndarray, np.ndarray]:
        sensor_df = self.load_sensor_data(motion_file)
        if sensor_df.empty:
            return np.array([]), np.array([])
        ann_file = self.find_annotation_file(motion_file, data_dir)
        ann_df = self.load_annotation_data(ann_file) if ann_file else pd.DataFrame()

        data = sensor_df[['accel_x', 'accel_y', 'accel_z']].values
        ts = sensor_df['timestamp'].values

        if window_type == 'small':
            wlen = self.small_window_len
            step = self.small_window_step
            label_func = self._get_existence_label
        else:
            wlen = self.big_window_len
            step = self.big_window_step
            label_func = self._get_window_label

        if len(data) < wlen:
            return np.array([]), np.array([])

        windows, labels = [], []
        for start_idx in range(0, len(data) - wlen + 1, step):
            end_idx = start_idx + wlen
            windows.append(data[start_idx:end_idx])
            labels.append(label_func(ts[start_idx], ts[end_idx-1], ann_df))
        return np.array(windows), np.array(labels)

    def create_windows(self, data_dir: Path, window_type='big') -> Tuple[np.ndarray, np.ndarray]:
        motion_dir = data_dir / "motion"
        if not motion_dir.exists():
            print(f"Motion dir missing: {motion_dir}")
            return np.array([]), np.array([])
        files = list(motion_dir.glob("*.csv")) or list(motion_dir.glob("m*.csv"))
        print(f"Found {len(files)} files in {data_dir.name}, type={window_type}")
        all_w, all_l = [], []
        for f in files:
            w, l = self.create_windows_from_file(f, data_dir, window_type)
            if len(w) > 0:
                all_w.append(w)
                all_l.append(l)
        if all_w:
            all_w = np.concatenate(all_w, axis=0)
            all_l = np.concatenate(all_l, axis=0)
            print(f"Total {len(all_w)} {window_type} windows")
            return all_w, all_l
        print(f"No windows for {data_dir.name}")
        return np.array([]), np.array([])

    def load_all_data(self, data_dirs: List[Path], window_type='big') -> Tuple[np.ndarray, np.ndarray]:
        all_w, all_l = [], []
        for d in data_dirs:
            if not d.exists():
                continue
            w, l = self.create_windows(d, window_type)
            if len(w) > 0:
                all_w.append(w)
                all_l.append(l)
        if all_w:
            all_w = np.concatenate(all_w, axis=0)
            all_l = np.concatenate(all_l, axis=0)
            print(f"Data summary ({window_type} windows): {len(all_w)} samples, shape {all_w.shape}")
            return all_w, all_l
        print(f"No data loaded for {window_type}")
        return np.array([]), np.array([])