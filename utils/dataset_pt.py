import os
import torch
from torch.utils.data import Dataset
import torchio as tio  # Medical image augmentation and patch sampling

class PT_Dataset(Dataset):
    def __init__(self, root, subset='train', transform=None):
        """
        Args:
            root (str): Root directory containing 'train', 'val', or 'test' folders
            subset (str): Dataset subset, one of ['train', 'val', 'test']
            transform (tio.Compose, optional): TorchIO transform to apply (e.g., augmentation)
        """
        self.root = os.path.join(root, subset)
        self.samples = sorted(os.listdir(self.root))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        sample_dir = os.path.join(self.root, sample_id)

        ct_path = os.path.join(sample_dir, f"{sample_id}_ct.pt")
        seg_path = os.path.join(sample_dir, f"{sample_id}_seg.pt")

        # Load tensors (from .pt files), ensure float type
        ct = torch.load(ct_path)
        seg = torch.load(seg_path)

        ct = ct.float()
        seg = seg.float()

        # Construct TorchIO Subject (required for patch sampling)
        subject = tio.Subject(
            ct=tio.ScalarImage(tensor=ct),
            seg=tio.LabelMap(tensor=seg),
            name=sample_id
        )

        # Apply optional TorchIO transforms (augmentation)
        if self.transform:
            subject = self.transform(subject)

        return subject
