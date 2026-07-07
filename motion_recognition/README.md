# 手机传感器动作识别系统

## 快速开始

### 方式1: 使用主脚本（推荐）
```bash
# 完整流程：训练 + 评估 + 演示
python scripts/main.py --mode all --epochs 50

# 仅训练
python scripts/main.py --mode train --model_type lightweight --epochs 50

# 仅评估
python scripts/main.py --mode evaluate

# 仅演示
python scripts/main.py --mode demo --demo_file motion_data/training_data/single_motion/motion/m11.csv

# 快速测试（20轮训练 + 快速评估）
python scripts/main.py --mode test