#!/bin/bash
#SBATCH --job-name=preprocess_nsclc_radgen
#SBATCH --output=preprocess_nsclc_radgen_logs/preprocess_nsclc_radgen_%j.out
#SBATCH --error=preprocess_nsclc_radgen_logs/preprocess_nsclc_radgen_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

# ==== 环境设置 ====
# module load python/3.11.9

# === 切换到项目目录 ===
cd /project2/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/radiogenomics  # ← 修改为你 main.py 所在目录

# === 自动安装 requirements.txt ===
# pip install --user --no-cache-dir -r requirements.txt
# ==== 训练命令 ====
python test.py