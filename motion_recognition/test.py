"""
测试实时系统完整流程
"""
import sys
import os
import numpy as np
import time
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from motion_recognition.inference.realtime_predictor import (
    add_data_and_predict, get_system_status, clear_system
)

def test_simulation():
    """测试模拟实时数据流"""
    print("=" * 60)
    print("测试模拟实时数据流")
    print("=" * 60)
    
    # 清空系统
    print("清空系统...")
    clear_result = clear_system()
    print(f"清空结果: {clear_result}")
    
    # 模拟不同动作的数据
    actions = ['right_down', 'left_down', 'circle_clock', 'swipe_up', 'swipe_down']
    
    for action_idx, action_name in enumerate(actions):
        print(f"\n模拟动作: {action_name}")
        
        # 生成模拟数据（模拟2秒数据）
        n_samples = 100  # 2秒数据
        t = np.arange(n_samples) / 50.0
        
        # 根据动作类型生成不同的加速度模式
        if action_name == 'circle_clock':
            # 顺时针圆圈：x和y相位差90度的正弦波
            acc_x = 5 * np.sin(2 * np.pi * 1.0 * t)
            acc_y = 5 * np.cos(2 * np.pi * 1.0 * t)
            acc_z = np.zeros_like(t)
        elif action_name == 'right_down':
            # 右下划：x和y同相的正弦波
            acc_x = 8 * np.sin(2 * np.pi * 1.5 * t)
            acc_y = 6 * np.sin(2 * np.pi * 1.5 * t)
            acc_z = np.zeros_like(t)
        elif action_name == 'left_down':
            # 左下划：x和y反相的正弦波
            acc_x = 8 * np.sin(2 * np.pi * 1.5 * t)
            acc_y = -6 * np.sin(2 * np.pi * 1.5 * t)
            acc_z = np.zeros_like(t)
        elif action_name == 'swipe_up':
            # 上划：主要是y轴变化
            acc_x = np.zeros_like(t)
            acc_y = 10 * np.sin(2 * np.pi * 2.0 * t)
            acc_z = np.zeros_like(t)
        elif action_name == 'swipe_down':
            # 下划：主要是y轴负向变化
            acc_x = np.zeros_like(t)
            acc_y = -10 * np.sin(2 * np.pi * 2.0 * t)
            acc_z = np.zeros_like(t)
        else:
            # 随机数据
            acc_x = np.random.randn(n_samples) * 2
            acc_y = np.random.randn(n_samples) * 2
            acc_z = np.random.randn(n_samples) * 2
        
        # 添加噪声
        noise = np.random.randn(n_samples, 3) * 0.3
        acc_data = np.column_stack([acc_x, acc_y, acc_z]) + noise
        
        # 分批发送数据，模拟实时流
        batch_size = 25  # 每次发送0.5秒数据
        for i in range(0, n_samples, batch_size):
            batch = acc_data[i:i+batch_size]
            if len(batch) > 0:
                # 添加时间戳
                batch_t = t[i:i+batch_size]
                sensor_data = np.column_stack([batch_t, batch])
                
                # 发送到预测器
                result = add_data_and_predict(sensor_data)
                
                # 显示结果
                if result.get('status') == 'success':
                    predictions = result.get('predictions', [])
                    if predictions:
                        pred = predictions[0]
                        detected_action = pred.get('action', 'unknown')
                        confidence = pred.get('confidence', 0)
                        
                        print(f"  批次 {i//batch_size+1}: {detected_action} ({confidence:.2f})")
                
                # 模拟实时延迟
                time.sleep(0.1)
        
        # 显示系统状态
        status = get_system_status()
        print(f"  系统状态: {status.get('current_action', '无')}")
        print(f"  总预测次数: {status.get('total_predictions', 0)}")
    
    print("\n" + "=" * 60)
    print("模拟测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    test_simulation()