#!/usr/bin/env python3
"""
Motion Recognition System - Main entry.
Supports training, evaluation, real‑time demo, and combined flows.
"""
import argparse
import sys
import numpy as np
import time
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from motion_recognition.config import config
from motion_recognition.data.dataset import MotionDataset
from motion_recognition.models.cnn_1d import create_small_detector, create_big_classifier
from motion_recognition.training.trainer import MotionTrainer
from motion_recognition.inference.realtime_predictor import RealtimePredictor

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def check_data_dirs(dirs):
    valid = []
    for d in dirs:
        if d.exists() and (d/"motion").exists() and list((d/"motion").glob("*.csv")):
            valid.append(d)
            print(f"✓ {d.name}: data found")
        else:
            print(f"✗ {d.name}: invalid or empty")
    return valid

def run_training(model_type="small", epochs=None, include=None, exclude=None):
    if epochs is None:
        epochs = config.EPOCHS
    print_section(f"Training {model_type} model")

    dataset = MotionDataset(config)
    all_dirs = config.get_train_data_dirs()
    valid = check_data_dirs(all_dirs)
    if not valid:
        print("No valid data directories")
        return False

    if include:
        incl = set(n.lower() for n in include)
        filtered = [d for d in valid if d.name.lower() in incl]
    elif exclude:
        excl = set(n.lower() for n in exclude)
        filtered = [d for d in valid if d.name.lower() not in excl]
    else:
        filtered = valid

    if not filtered:
        print("No directories after filtering")
        return False
    print(f"Using: {[d.name for d in filtered]}")

    windows, labels = dataset.load_all_data(filtered, window_type=model_type)
    if len(windows) == 0:
        print(f"No {model_type} data loaded")
        return False
    print(f"Loaded {len(windows)} samples, shape {windows.shape}")

    builder = create_small_detector(config) if model_type=='small' else create_big_classifier(config)
    trainer = MotionTrainer(config, builder, model_type)
    start = time.time()
    trainer.train(windows, labels, epochs=epochs, batch_size=config.BATCH_SIZE)
    print(f"Training finished in {time.time()-start:.1f}s")
    builder.save_model()
    trainer.save_training_history()
    return True

def run_normal_evaluation(model_type="small"):
    print_section(f"Normal evaluation ({model_type})")
    builder = create_small_detector(config) if model_type=='small' else create_big_classifier(config)
    model_path = config.get_small_model_path() if model_type=='small' else config.get_big_model_path()
    if not model_path.exists():
        print(f"Model missing: {model_path}")
        return False
    builder.load_model(model_path)
    model = builder.model
    print("Model loaded")

    dataset = MotionDataset(config)
    val_dirs = [config.VAL_DATA_PATH] if config.VAL_DATA_PATH.exists() else []
    if not val_dirs:
        val_dirs = [d for d in config.get_train_data_dirs() if d.exists()][:1]
    if not val_dirs:
        print("No evaluation data")
        return False

    motion_root = val_dirs[0] / "motion"
    if not motion_root.exists():
        print(f"Motion directory missing: {motion_root}")
        return False
    motion_files = list(motion_root.glob("*.csv"))
    if not motion_files:
        motion_files = []
        for sub in motion_root.iterdir():
            if sub.is_dir() and sub.name.lower() != 'compare':
                motion_files.extend(sub.glob("*.csv"))
    if not motion_files:
        print("No evaluation files")
        return False
    print(f"Found {len(motion_files)} files")

    print("\n" + "-"*80)
    print(f"{'File':<20} {'Loss':<10} {'Accuracy':<10} {'Samples':<10}")
    print("-"*80)

    total_loss, total_acc, cnt = 0,0,0
    for mf in sorted(motion_files):
        stem = mf.stem
        ann = config.VAL_DATA_PATH / "annotation" / f"{stem}.csv"
        if not ann.exists() and stem.startswith('m'):
            ann = config.VAL_DATA_PATH / "annotation" / f"a{stem[1:]}.csv"
        if not ann.exists():
            print(f"⚠ Annotation missing for {mf.name}, skip")
            continue
        windows, labels = dataset.create_windows_from_file(mf, ann.parent.parent, window_type=model_type)
        if len(windows)==0:
            continue
        loss, acc = model.evaluate(windows, labels, verbose=0)
        total_loss += loss; total_acc += acc; cnt += 1
        print(f"{mf.name:<20} {loss:<10.4f} {acc:<10.4f} {len(windows):<10}")
    print("-"*80)
    if cnt:
        print(f"\nSummary over {cnt} files:")
        print(f"  Average Loss: {total_loss/cnt:.4f}")
        print(f"  Average Accuracy: {total_acc/cnt:.4f}")
    else:
        print("No files evaluated")
    return True

def run_compare_evaluation():
    print_section("Compare evaluation (per‑action top3)")
    builder = create_big_classifier(config)
    model_path = config.get_big_model_path()
    if not model_path.exists():
        print(f"Model missing: {model_path}")
        return False
    builder.load_model(model_path)
    model = builder.model

    compare_motion = config.VAL_DATA_PATH / "motion" / "compare"
    compare_anno = config.VAL_DATA_PATH / "annotation" / "compare"
    if not compare_motion.exists() or not compare_anno.exists():
        print("Compare directories not found")
        return False

    motion_files = list(compare_motion.glob("*.csv"))
    if not motion_files:
        print("No CSV files in compare/motion")
        return False
    print(f"Found {len(motion_files)} files")

    dataset = MotionDataset(config)
    table_rows = []

    def find_anno(f):
        stem = f.stem
        cand = compare_anno / f"{stem}.csv"
        if cand.exists(): return cand
        if stem.startswith('m'):
            cand = compare_anno / f"a{stem[1:]}.csv"
            if cand.exists(): return cand
        if stem.startswith('cm'):
            cand = compare_anno / f"{stem[1:]}.csv"
            if cand.exists(): return cand
        for suf in ['_label','_annotation']:
            cand = compare_anno / f"{stem}{suf}.csv"
            if cand.exists(): return cand
        return None

    def eval_file(mf, af):
        sensor = dataset.load_sensor_data(mf)
        if sensor.empty: return []
        anno = dataset.load_annotation_data(af)
        if anno.empty: return []
        windows, _ = dataset.create_windows_from_file(mf, af.parent.parent, window_type='big')
        if len(windows)==0: return []
        preds = model.predict(windows, verbose=0)
        ts = sensor['timestamp'].values
        wlen = config.get_big_window_length_samples()
        step = config.get_big_window_step_samples()
        data = sensor[['accel_x','accel_y','accel_z']].values
        win_times = []
        for i in range(0, len(data)-wlen+1, step):
            win_times.append((ts[i], ts[i+wlen-1]))
        if len(win_times) != len(preds):
            m = min(len(win_times), len(preds))
            win_times = win_times[:m]
            preds = preds[:m]
        results = []
        for _, row in anno.iterrows():
            true_label = row['action_label']
            start, end = row['start_time'], row['end_time']
            relevant = [i for i,(ws,we) in enumerate(win_times) if we>start and ws<end]
            if not relevant: continue
            avg = np.mean(preds[relevant], axis=0)
            top3_idx = np.argsort(avg)[-3:][::-1]
            top3 = [(config.ACTION_CLASSES[i], avg[i]) for i in top3_idx]
            top1 = top3[0] if len(top3)>0 else ('none',0)
            top2 = top3[1] if len(top3)>1 else ('none',0)
            top3v = top3[2] if len(top3)>2 else ('none',0)
            results.append((true_label, top1, top2, top3v))
        return results

    for mf in sorted(motion_files):
        af = find_anno(mf)
        if af is None:
            print(f"⚠ No annotation for {mf.name}, skip")
            continue
        res = eval_file(mf, af)
        if not res:
            print(f"No actions in {mf.name}")
            continue
        for true, t1, t2, t3 in res:
            table_rows.append({
                'File': mf.name,
                'True Label': true,
                'Top1 Label': t1[0], 'Top1 Conf': t1[1],
                'Top2 Label': t2[0], 'Top2 Conf': t2[1],
                'Top3 Label': t3[0], 'Top3 Conf': t3[1]
            })

    if table_rows:
        df = pd.DataFrame(table_rows)
        for col in ['Top1 Conf','Top2 Conf','Top3 Conf']:
            df[col] = df[col].map('{:.3f}'.format)
        # Insert separators between files
        new_rows = []
        cur = None
        for _, row in df.iterrows():
            f = row['File']
            if cur is not None and f != cur:
                new_rows.append({c: '-' for c in df.columns})
            new_rows.append(row.to_dict())
            cur = f
        df_sep = pd.DataFrame(new_rows)
        pd.set_option('display.max_colwidth', 30)
        pd.set_option('display.width', 180)
        pd.set_option('display.colheader_justify', 'center')
        print("\nComparison Results Table:")
        print(df_sep.to_string(index=False))
    else:
        print("No results")
    return True

def run_realtime_demo(duration=30, data_file=None):
    print_section("Real-time demo")
    try:
        predictor = RealtimePredictor(config)
    except Exception as e:
        print(f"Failed to init predictor: {e}")
        return False

    if data_file is None:
        val_motion = config.VAL_DATA_PATH / "motion"
        if val_motion.exists():
            files = sorted(val_motion.glob("*.csv"))
            if files:
                data_file = str(files[0])
                print(f"Auto‑using: {Path(data_file).name}")
            else:
                print("No CSV in validation set, falling back to simulated")
        else:
            print("Validation set missing, using simulated data")

    if data_file:
        path = Path(data_file)
        if not path.exists():
            print(f"File not found: {data_file}")
            return False
        dataset = MotionDataset(config)
        df = dataset.load_sensor_data(path)
        if df.empty:
            print("Cannot load sensor data")
            return False
        acc = df[['accel_x','accel_y','accel_z']].values
        ts = df['timestamp'].values
        print(f"Loaded {len(acc)} samples, playing back... Press Ctrl+C to stop")
        try:
            for i in range(len(acc)):
                sample = acc[i].reshape(1,3)
                res = predictor.process_incremental(sample)
                if i % 5 == 0:
                    t = ts[i] if i<len(ts) else i*0.1
                    print(f"t={t:.1f}s: action={res['action']}, conf={res['confidence']:.2f}, has_action={res['action_present']}")
                    if hasattr(predictor,'big_pred_history') and predictor.big_pred_history:
                        latest = predictor.big_pred_history[-1]
                        top3 = np.argsort(latest)[-3:][::-1]
                        top3_str = ", ".join([f"{predictor.config.get_action_label(i)}:{latest[i]:.2f}" for i in top3])
                        print(f"    top3: {top3_str}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nInterrupted")
        print("Playback finished")
        return True
    else:
        # Simulated data mode (keep original)
        print("Simulated data stream...")
        sample_interval = 0.1
        total = 0
        try:
            while total < duration:
                t = total
                phase = int(t/5)%5
                if phase==0:
                    acc = np.random.randn(3)*0.3; acc[2]+=9.8
                elif phase==1:
                    acc = np.array([5*np.sin(2*np.pi*1*t), -5*np.sin(2*np.pi*1*t), 9.8])
                elif phase==2:
                    acc = np.array([5*np.sin(2*np.pi*0.8*t), 5*np.cos(2*np.pi*0.8*t), 9.8])
                elif phase==3:
                    acc = np.array([0, 8*np.sin(2*np.pi*1*t), 9.8])
                else:
                    acc = np.array([0, -8*np.sin(2*np.pi*1*t), 9.8])
                acc += np.random.randn(3)*0.2
                res = predictor.process_incremental(acc.reshape(1,3))
                if int(total*2)%1==0:
                    print(f"t={total:.1f}s: action={res['action']}, conf={res['confidence']:.2f}, has_action={res['action_present']}")
                    if hasattr(predictor,'big_pred_history') and predictor.big_pred_history:
                        latest = predictor.big_pred_history[-1]
                        top3 = np.argsort(latest)[-3:][::-1]
                        top3_str = ", ".join([f"{predictor.config.get_action_label(i)}:{latest[i]:.2f}" for i in top3])
                        print(f"    top3: {top3_str}")
                time.sleep(sample_interval)
                total += sample_interval
        except KeyboardInterrupt:
            print("\nInterrupted")
        print("Demo finished")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','evaluate','realtime','all','compare'], default='train')
    parser.add_argument('--model_type', choices=['small','big'], default=None)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--include', nargs='+', default=None)
    parser.add_argument('--exclude', nargs='+', default=None)
    parser.add_argument('--compare', action='store_true', help='Use compare evaluation (per-action top3)')
    args = parser.parse_args()

    print("="*60)
    print("Motion Recognition System")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.mode in ['train','all']:
        if args.model_type:
            print(f"Model type: {args.model_type}")
        print(f"Epochs: {args.epochs}")
        if args.include:
            print(f"Include: {args.include}")
        elif args.exclude:
            print(f"Exclude: {args.exclude}")
        else:
            print("Using all training directories")
    if args.mode in ['realtime','all']:
        if args.file:
            print(f"Data file: {args.file}")
        else:
            print("Data file: auto from validation set")

    success = True
    try:
        if args.mode == 'train':
            if args.model_type is None:
                success &= run_training('small', args.epochs, args.include, args.exclude)
                success &= run_training('big', args.epochs, args.include, args.exclude)
            else:
                success = run_training(args.model_type, args.epochs, args.include, args.exclude)
        elif args.mode == 'evaluate':
            if args.compare:
                # Compare mode always uses big classifier
                success = run_compare_evaluation()
            else:
                if args.model_type is None:
                    # Default: evaluate both small and big
                    print("Evaluating both small and big models (default). Use --model_type to evaluate only one.")
                    success = True
                    success &= run_normal_evaluation('small')
                    success &= run_normal_evaluation('big')
                else:
                    # Evaluate only specified model
                    success = run_normal_evaluation(args.model_type)
        elif args.mode == 'realtime':
            success = run_realtime_demo(args.duration, args.file)
        elif args.mode == 'all':
            success = True
            success &= run_training('small', args.epochs, args.include, args.exclude)
            success &= run_training('big', args.epochs, args.include, args.exclude)
            success &= run_normal_evaluation('small')
            success &= run_normal_evaluation('big')
            success &= run_realtime_demo(args.duration, args.file)
        elif args.mode == 'compare':
            success = run_training('big', args.epochs, args.include, args.exclude)
            if success:
                success = run_compare_evaluation()
        if success:
            print("\n✓ All operations completed successfully")
        else:
            print("\n⚠ Some operations failed")
    except KeyboardInterrupt:
        print("\nInterrupted")
        success = False
    except Exception as e:
        print(f"\nError: {e}")
        import traceback; traceback.print_exc()
        success = False
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()