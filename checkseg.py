import os
import shutil
import random

# 源数据路径（已筛选好的 lung_roi 数据）
source_dir = "../datasets_XW/processed_nsclc_radiogenomics/lungs_roi"

# 目标路径
target_root = "../datasets_XW/split/RADGEN"
train_dir = os.path.join(target_root, "train")
test_dir = os.path.join(target_root, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有病患目录
patients = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

# 打乱并划分
random.seed(42)
random.shuffle(patients)

split_idx = int(0.8 * len(patients))
train_patients = patients[:split_idx]
test_patients = patients[split_idx:]

# 复制函数
def copy_patient(patient_id, dst_dir):
    src = os.path.join(source_dir, patient_id)
    dst = os.path.join(dst_dir, patient_id)
    shutil.copytree(src, dst)

# 执行复制
for pid in train_patients:
    copy_patient(pid, train_dir)
for pid in test_patients:
    copy_patient(pid, test_dir)

print(f"✅ 训练集: {len(train_patients)} 例")
print(f"✅ 测试集: {len(test_patients)} 例")
print(f"📁 输出路径: {target_root}/train 和 {target_root}/test")
