#!/bin/bash
#SBATCH --job-name=train_nsclc
#SBATCH --output=nsclc_logs/train_nsclc_%j.out
#SBATCH --error=nsclc_logs/train_nsclc_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

# ==== 环境设置 ====
# module load python/3.11.9

# === 切换到项目目录 ===
cd /project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/radiogenomics  # ← 修改为你 main.py 所在目录

# === 自动安装 requirements.txt ===
# pip install --user --no-cache-dir -r requirements.txt
# ==== 训练命令 ====
python main.py train_seg ../datasets_XW/split1/NSCLC/ results/NSCLC NSCLC \
  --batch_size 1 --epochs 100
