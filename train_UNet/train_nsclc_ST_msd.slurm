#!/bin/bash
#SBATCH --job-name=train_nsclc_ST_msd
#SBATCH --output=nsclc_ST_msd_logs/train_nsclc_ST_msd_%j.out
#SBATCH --error=nsclc_ST_msd_logs/train_nsclc_ST_msd_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

# ==== 环境设置 ====
# module load python/3.11.9

# === 切换到项目目录 ===
cd /project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/radiogenomics  # ← 修改为你 main.py 所在目录

# === 自动安装 requirements.txt ===
# pip install --user --no-cache-dir -r requirements.txt
# ==== 训练命令 ====
python main.py train_seg ../datasets_XW/split/NSCLC+ST_MSD/ results/NSCLC_ST_MSD NSCLC_ST_MSD \
  --batch_size 1 --epochs 100
