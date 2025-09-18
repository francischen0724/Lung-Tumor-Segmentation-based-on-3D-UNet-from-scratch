import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import torchio as tio
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from models.ra_seg import RA_Seg  # 替换为 RA_Seg
from torch.optim.lr_scheduler import ReduceLROnPlateau

def load_subjects(data_dir):
    subject_list = []
    sample_dirs = sorted(os.listdir(data_dir))
    for sample_id in sample_dirs:
        ct_path = os.path.join(data_dir, sample_id, f"{sample_id}_ct.pt")
        seg_path = os.path.join(data_dir, sample_id, f"{sample_id}_seg.pt")
        bbx_path = os.path.join(data_dir, sample_id, f"{sample_id}_bbx.pt")
        if not os.path.exists(ct_path) or not os.path.exists(seg_path) or not os.path.exists(bbx_path):
            continue
        subject = tio.Subject(
            ct=tio.ScalarImage(tensor=torch.load(ct_path)),
            seg=tio.LabelMap(tensor=torch.load(seg_path)),
            bbx=tio.LabelMap(tensor=torch.load(bbx_path))
        )
        subject_list.append(subject)
    return subject_list

def compute_iou(pred, target):
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    intersection = ((pred == 1) & (target == 1)).sum()
    union = ((pred == 1) | (target == 1)).sum()
    return intersection / union if union != 0 else 0.0

def train_seg(args):
    data_path = args.input
    output_path = args.output
    model_class = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device, flush=True)

    train_subjects = load_subjects(os.path.join(data_path, 'train'))
    val_subjects = load_subjects(os.path.join(data_path, 'test'))

    train_transform = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=0, p=0.3),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=5, p=0.3),
        tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=(1.0, 1.5), p=0.2)
    ])

    train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
    val_dataset = tio.SubjectsDataset(val_subjects)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = RA_Seg(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
    ).to(device)

    criterion = DiceCELoss(sigmoid=True, include_background=False, reduction="mean")
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=True)

    class EarlyStopping:
        def __init__(self, patience=10, delta=0.001):
            self.patience = patience
            self.counter = 0
            self.best_score = None
            self.delta = delta
            self.early_stop = False

        def __call__(self, val_score):
            if self.best_score is None:
                self.best_score = val_score
            elif val_score < self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0

    early_stopper = EarlyStopping(patience=10, delta=0.001)
    best_dice = 0.0

    train_losses, val_losses, val_dices, val_ious = [], [], [], []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)

        model.train()
        running_train_loss = 0.0
        skipped_batches = 0

        for batch in tqdm(train_loader, desc="Training"):
            ct = batch['ct'][tio.DATA].to(device)
            seg = batch['seg'][tio.DATA].to(device)
            bbx = batch['bbx'][tio.DATA].to(device).float()  # ✅ 强制转 float32

            # ✅ 添加 bbx 值域断言
            assert bbx.min() >= 0 and bbx.max() <= 1, "❌ bbx values not in [0,1]"

            # ✅ 添加维度断言
            assert bbx.shape[1] == 1, f"❌ bbx channel should be 1, got {bbx.shape[1]}"

            if seg.sum() / seg.numel() < 0.001:
                skipped_batches += 1
                continue

            pred, _ = model(ct, bbx)
            loss = criterion(pred, seg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * ct.size(0)

        print(f"Skipped batches due to empty seg: {skipped_batches}", flush=True)  # ✅ 输出跳过数量

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        dice_metric.reset()
        iou_scores = []

        with torch.no_grad():
            for subject in tqdm(val_loader, desc="Validating"):
                ct = subject['ct'][tio.DATA].to(device)
                seg = subject['seg'][tio.DATA].to(device)
                bbx = subject['bbx'][tio.DATA].to(device).float()  # ✅ val 中也强制 float32

                pred, _ = model(ct, bbx)
                loss = criterion(pred, seg)
                running_val_loss += loss.item() * ct.size(0)

                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                dice_metric(pred_bin, seg)
                iou_scores.append(compute_iou(pred_bin, seg))

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_dice = dice_metric.aggregate()[0].item()
        avg_val_iou = sum(iou_scores) / len(iou_scores)

        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        val_ious.append(avg_val_iou)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}", flush=True)

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(output_path, f"{model_class}_best.pt"))
            print("✅ Best model updated and saved.", flush=True)

        scheduler.step(avg_val_dice)
        early_stopper(avg_val_dice)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered. Training halted.", flush=True)
            break

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, f"{model_class}_final.pt"))

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training/Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{model_class}_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(val_dices, label='Val Dice')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{model_class}_dice_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(val_ious, label='Val IoU')
    plt.title('Validation IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(output_path, f"{model_class}_iou_curve.png"))
    plt.close()
