import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess_original import Preprocess
from experiments.inf_seg import Inference
from models.ra_seg import RA_Seg
from models.unet import UNet

MODELS = {
    "UNet": UNet,
    "RA_Seg": RA_Seg,
}

# Preprocess

# # NSCLC
# input_path = "/project/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/datasets_XW/NSCLC-Radiomics/extracted_all"
# output_path = "../datasets_XW/processed_nsclc1"
# pre = Preprocess(datapath=input_path, dataoutput=output_path, dataset='rad')
# pre.radiomics()

# # MSD
# input_path = "/project/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/datasets_XW/Task06_Lung"
# output_path = "../datasets_XW/processed_msd1"

# pre = Preprocess(datapath=input_path, dataoutput=output_path, dataset='msd')
# pre.msd()

# # NSCLC Radiogenomics
# input_path = "/project/ruishanl_1185/Tumor_Segmentation_Summer2025_XWDR/Xiwen/datasets_XW/NSCLC-Radiogenomics/"
# output_path = "../datasets_XW/processed_nsclc_radiogenomics"

# pre = Preprocess(datapath=input_path, dataoutput=output_path, dataset='radgen')
# pre.radiogenomics()

# Inference
data_path = "../datasets_XW/split/RADGEN_NSCLC/test/"
weights_path = "./results_UNet/RADGEN/RADGEN_best.pt"
model_class = "UNet"  # or "RA_Seg"

infer = Inference(data_path, weights_path, model_class)
infer.run()




# # 尝试加载
# weights = torch.load("weights/radgen_finetune.pt", map_location="cpu")
# print(type(weights))
