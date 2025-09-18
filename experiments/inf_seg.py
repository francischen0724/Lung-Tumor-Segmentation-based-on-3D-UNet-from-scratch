# Imports
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.utils import data_init_inf
from models.unet import UNet
from models.ra_seg import RA_Seg
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import matplotlib.pyplot as plt


MODELS ={
    "UNet": UNet,
    "RA_Seg": RA_Seg
}

CRITERIONS = {
    "CE": nn.CrossEntropyLoss,
    "BCE": nn.BCELoss,
    "BCEL": nn.BCEWithLogitsLoss,
    "FOCAL": FocalLoss,
    "DICE": DiceLoss,
    "DICE_CE": DiceCELoss,
}

METRIC = {
    "DICE": DiceMetric,
}

def compute_iou(pred, target):
    pred = pred.view(-1).cpu().numpy()
    target = target.view(-1).cpu().numpy()
    intersection = ((pred == 1) & (target == 1)).sum()
    union = ((pred == 1) | (target == 1)).sum()
    return intersection / union if union != 0 else 0.0

class Inference:
    def __init__(self, data_path, weights_path, model_class):
        self.data_path = data_path
        self.weights_path = weights_path
        self.model_class = model_class

        self.device = torch.device(
            "cuda:" + str(torch.cuda.current_device())
            if torch.cuda.is_available()
            else "cpu"
        )

        if torch.cuda.is_available():
            print("Running on GPU:", torch.cuda.current_device())
        else:
            print("Running on CPU")

        try:
            self.model = MODELS[model_class](
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=[64, 128, 256, 512],
                strides=[2, 2, 2],
            ).to(self.device)
            print("Model {} loaded".format(model_class))
        except:
            print("Model {} not found".format(model_class))
            print(sorted(MODELS))
            exit()
        # try:
        #     self.model = MODELS[model_class](
        #         spatial_dims=3,
        #         in_channels=1,
        #         out_channels=1,
        #         channels=[64, 128, 256, 512, 1024],  # 修正这里！
        #         strides=[2, 2, 2, 2],                # 修正这里！
        #     ).to(self.device)
        #     print("✅ Model {} loaded".format(model_class))
        # except Exception as e:
        #     print(f"❌ Error loading model {model_class}: {e}")
        #     print("Available models:", sorted(MODELS))
        #     exit()


        self.model.load_state_dict(torch.load(self.weights_path))
        print("Weights loaded from {}".format(self.weights_path))

        self.testset = data_init_inf.DatasetInit(
            path=self.data_path,
            subset="test",
            channels=[64, 128, 256, 512],
            mode="vanilla",
        )

        self.test_loader = DataLoader(
            self.testset,
            batch_size=1,
            num_workers=4,
            shuffle=False
        )

        self.criterion = CRITERIONS["DICE_CE"](to_onehot_y=True)
        self.metric = METRIC["DICE"](include_background=False, reduction="mean")

    def post_process(self, img):
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        img = img.to(torch.int64)
        return img

    def feat_preprocess(self, features, patient):
        vector = features.mean(dim=(-2, -1)).flatten(start_dim=1)
        vector = vector.squeeze().cpu().detach().numpy()
        np.save(os.path.join(self.data_path, patient, "{}_raw_feat.npy".format(patient)), vector)
        print('Raw high-Level features extracted.')
        return

    def save_seg(self, seg, patient):
        seg = seg.squeeze().cpu().detach().numpy()
        np.save(os.path.join(self.data_path, patient, "{}_seg.npy".format(patient)), seg)
        return

    def run(self):

        self.model.eval()
        dice_list = []
        iou_list = []

        print("Starting inference...")
        with torch.no_grad():
            for batch_num, data in enumerate(tqdm(self.test_loader, desc="Testing")):
                inp, gt, patient = data
                inp, gt = inp.to(self.device), gt.to(self.device)

                if self.model_class == "RA_Seg":
                    organ = gt  # 注意：若有 organ 输入，请修改此行为 organ 输入
                    test_output, hl_feat = self.model(inp, organ)
                else:
                    test_output = self.model(inp)

                test_output = torch.sigmoid(test_output)
                test_output_bin = self.post_process(test_output)

                self.metric.reset()
                self.metric(test_output_bin, gt)
                dice_score = self.metric.aggregate()[0].item()
                dice_list.append(dice_score)

                iou_score = compute_iou(test_output_bin, gt)
                iou_list.append(iou_score)

                self.save_seg(test_output_bin, patient[0])

        print("\n📊 Inference Results on Test Set:")
        print(f"✅ Average Dice: {np.mean(dice_list):.4f}")
        print(f"✅ Average IoU : {np.mean(iou_list):.4f}")

        # # 可视化结果保存路径
        # vis_dir = os.path.join(self.data_path, "results_viz")
        # os.makedirs(vis_dir, exist_ok=True)

        # # Dice 曲线图
        # plt.figure()
        # plt.plot(dice_list, marker='o', label='Dice')
        # plt.title("Dice Score per Sample")
        # plt.xlabel("Sample Index")
        # plt.ylabel("Dice Score")
        # plt.ylim(0, 1)
        # plt.grid()
        # plt.legend()
        # plt.savefig(os.path.join(vis_dir, "dice_curve.png"))
        # plt.close()

        # # IoU 曲线图
        # plt.figure()
        # plt.plot(iou_list, marker='x', color='orange', label='IoU')
        # plt.title("IoU Score per Sample")
        # plt.xlabel("Sample Index")
        # plt.ylabel("IoU Score")
        # plt.ylim(0, 1)
        # plt.grid()
        # plt.legend()
        # plt.savefig(os.path.join(vis_dir, "iou_curve.png"))
        # plt.close()


