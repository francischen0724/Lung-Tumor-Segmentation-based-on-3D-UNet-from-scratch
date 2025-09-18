import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置路径
patient_id = "lung_001"

# ct_path = f"../datasets_XW/processed_nsclc_radiogenomics/full_ct/{patient_id}/{patient_id}_ct.pt"
# seg_path = f"../datasets_XW/processed_nsclc_radiogenomics/full_ct/{patient_id}/{patient_id}_seg.pt"

ct_path = f"../datasets_XW/processed_msd/lungs_roi/{patient_id}/{patient_id}_ct.pt"
seg_path = f"../datasets_XW/processed_msd/lungs_roi/{patient_id}/{patient_id}_seg.pt"
save_dir = "./visualizations"
os.makedirs(save_dir, exist_ok=True)

# 读取数据
ct = torch.load(ct_path).squeeze()
seg = torch.load(seg_path).squeeze()
# seg = torch.load(seg_path).tensor.squeeze()

# 选择肿瘤像素最多的切片
tumor_sums = seg.sum(axis=(1, 2))
if (tumor_sums > 0).any():
    slice_idx = int(np.argmax(tumor_sums))
else:
    slice_idx = ct.shape[0] // 2  # fallback 中间切片

# 获取该切片的图像和标签
ct_slice = ct[slice_idx]
seg_slice = seg[slice_idx]

# 可视化三联图
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(ct_slice, cmap='gray')
axs[0].set_title("CT Slice")
axs[0].axis('off')

axs[1].imshow(seg_slice, cmap='Reds')
axs[1].set_title("GT Tumor")
axs[1].axis('off')

axs[2].imshow(ct_slice, cmap='gray')
axs[2].imshow(seg_slice, cmap='Reds', alpha=0.5)
axs[2].set_title("Overlay")
axs[2].axis('off')

# 保存
out_path = os.path.join(save_dir, f"{patient_id}_tumor_max_slice.png")
plt.tight_layout()
plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"✅ Saved maximum tumor slice visualization to: {out_path}")
