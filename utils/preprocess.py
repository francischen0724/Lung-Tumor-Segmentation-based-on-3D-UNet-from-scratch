"""
Author: Ivo Gollini Navarrete
Date: 21/august/2022
Institution: MBZUAI
"""

# IMPORTS
import os
import numpy as np
# import pandas as pd
import SimpleITK as sitk
import pydicom as dicom
import nibabel as nib

import torch
import torchio.transforms as tt
from torchvision.ops import masks_to_boxes
from lungmask import mask

# from tqdm import tqdm

class Preprocess:
    """
    Prepare data for experiments.
    """
    def __init__(self, datapath, dataoutput, dataset):
        self.datapath = datapath
        self.dataoutput = dataoutput
        self.dataset = dataset

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print('GPU Available')
        else: 
            self.device = torch.device("cpu")
            print("GPU Not Available ")

    def msd(self):
        print('📦 Preprocessing MSD dataset...')
        ct_dir = os.path.join(self.datapath, 'imagesTr')
        seg_dir = os.path.join(self.datapath, 'labelsTr')
        
        ct_list = sorted(os.listdir(ct_dir))
        seg_list = sorted(os.listdir(seg_dir))
        
        assert len(ct_list) == len(seg_list), "Mismatch between CT and label counts"
        
        for pat, label in zip(ct_list, seg_list):
            try:
                # 保证 patient ID 统一
                patient = pat.replace('.nii.gz', '').split('_')[-1]
                print(f"\n🧠 Processing patient {patient}")
                
                # --- Load CT ---
                ct_path = os.path.join(ct_dir, pat)
                ct_img = nib.load(ct_path)
                ct = self.reorient(ct_img.get_fdata())

                # --- Load Seg ---
                seg_path = os.path.join(seg_dir, label)
                seg_img = nib.load(seg_path)
                seg = self.reorient(seg_img.get_fdata())

                assert ct.shape == seg.shape, f"Shape mismatch in {patient}"

                # --- Normalize & Save ---
                pre_ct = self.normalize(ct[None, :, :, :])
                pre_seg = self.normalize(seg[None, :, :, :], is_label=True)

                self.save_img(patient, pre_ct)
                self.save_img(patient, pre_seg, is_Label=True)
                print("✅ Normalized & saved")

                # --- Extract Lungs ---
                self.extract_lungs(patient, ct, seg[None, :, :, :])
                print("🫁 Lung ROI extracted")

            except Exception as e:
                print(f"❌ Failed processing {pat}: {e}")
                continue

        # --- Bounding box generation ---
        lungs_roi_dir = os.path.join(self.dataoutput, 'lungs_roi')
        self.tumor_bbx(lungs_roi_dir)
        print("🎯 Tumor bounding box saved\n")
        return

    def reorient(self, array):
        """统一将 NIfTI 图像转为 D, H, W 并修正方向"""
        array = np.transpose(array, (2, 1, 0))  # Z, Y, X
        array = np.flip(array, axis=1).copy()  # flip Y for consistency
        return array


    
    def radiomics(self):
        print('Preprocessing Radiomics dataset...')

        for patient in sorted(os.listdir(self.datapath)):
            patient_path = os.path.join(self.datapath, patient)
            if not os.path.isdir(patient_path): continue
            print(f"\n🩺 Processing {patient}")

            # 只处理每个病人目录下的第一个 study（假设只有一个）
            study_dirs = sorted([d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))])
            if not study_dirs:
                print(f"❌ No study directory found in {patient}")
                continue

            study_path = os.path.join(patient_path, study_dirs[0])
            ct_dir = os.path.join(study_path, 'CT')
            seg_dir = os.path.join(study_path, 'SEG')

            # --- 处理 CT ---
            if not os.path.exists(ct_dir) or len(os.listdir(ct_dir)) <= 1:
                print(f"⚠️ Skipping missing or empty CT folder: {ct_dir}")
                continue

            patient_ct = self.read_ct(ct_dir)
            patient_ct_np = sitk.GetArrayFromImage(patient_ct)
            preprocessed_ct = self.normalize(patient_ct_np[None, :, :, :])  # Add channel dim
            self.save_img(patient, preprocessed_ct)

            # --- 处理 SEG ---
            seg_path = os.path.join(seg_dir, '1-1.dcm')
            if not os.path.exists(seg_path):
                print(f"❌ Segmentation file missing: {seg_path}")
                continue

            seg_data = dicom.dcmread(seg_path)
            seg_array = seg_data.pixel_array
            seg_num = len(seg_data.SegmentSequence)
            print(f"🧠 Seg shape: {seg_array.shape}, with {seg_num} segmentations")

            # 找到 primary neoplasm 的索引
            seg_idx = 0
            for i in range(seg_num):
                label = seg_data.SegmentSequence[i].SegmentLabel
                if label == "Neoplasm, Primary":
                    print(f"✅ Primary Neoplasm in slice {i}")
                    seg_idx = i
                    break

            dim0 = int(seg_array.shape[0] / seg_num)
            seg_tensor = torch.reshape(torch.from_numpy(seg_array), (seg_num, dim0, 512, 512))
            patient_seg = seg_tensor[seg_idx].unsqueeze(0).numpy()

            preprocessed_seg = self.normalize(patient_seg, is_label=True)
            self.save_img(patient, preprocessed_seg, is_Label=True)

            # --- 提取肺部 ROI ---
            self.extract_lungs(patient, patient_ct_np, patient_seg)

    def radiogenomics(self):

        print('Preprocessing Radiogenomics dataset...')

        patients_list =  sorted(os.listdir(self.datapath))
        for patient in patients_list:
            print(patient)

            patient_path = os.path.join(self.datapath, patient, 'CT')
            patient_ct = self.read_ct(patient_path)
            patient_ct = sitk.GetArrayFromImage(patient_ct)
            preprocessed_ct = self.normalize(patient_ct[None, :, :, :])
            self.save_img(patient, preprocessed_ct)

            if os.path.exists(os.path.join(self.datapath, patient, 'seg')):
                patient_seg = self.read_ct(os.path.join(self.datapath, patient, 'seg'))
                patient_seg = sitk.GetArrayFromImage(patient_seg)
                
                preprocessed_seg = self.normalize(patient_seg, is_label=True)
                self.save_img(patient, preprocessed_seg, is_Label=True)

                self.extract_lungs(patient, patient_ct, patient_seg)
                self.extract_tumor(patient, patient_ct, torch.tensor(patient_seg[0]))
            
            else:
                self.extract_lungs(patient, patient_ct)
                continue
        self.tumor_bbx(self.dataoutput+'/lungs_roi')

    def extract_tumor(self, patient, ct, seg=None):
        tcr = self.roi_coord(seg, roi='tumor') # tcr = tumor_roi_coord
        tumor_ct = ct.copy()
        tumor_ct = self.crop_coord(tcr, tumor_ct)
        tumor_ct = self.normalize(tumor_ct, out_shape=(64, 64, 64))
        self.save_img(patient, tumor_ct, mode='tumor_roi')
    
    def extract_lungs(self, patient, ct, seg=None):
        print(f"🫁 Extracting lung mask for {patient} using lungmask R231 model")
        model_path = os.path.join(os.path.dirname(__file__), "R231.pth")
        model = mask.get_model('unet',modelpath=model_path).to(self.device)
        extracted = mask.apply(ct, model)
        extracted = torch.tensor(extracted)

        lungs_mask = extracted.clone()
        lcr = self.roi_coord(lungs_mask, roi='lungs')
        lungs_ct = ct.copy()
        lungs_ct = self.crop_coord(lcr, lungs_ct)
        lungs_ct = self.normalize(lungs_ct)
        self.save_img(patient, lungs_ct, mode='lungs_roi')

        if seg is not None:
            lungs_seg = seg.copy()
            lungs_seg = self.crop_coord(lcr, lungs_seg, is_label=True)
            lungs_seg = self.normalize(lungs_seg, is_label=True)
            self.save_img(patient, lungs_seg, mode='lungs_roi', is_Label=True)

    
    def crop_coord(self, coord, image, is_label=False):
        if is_label:
            image = image[:, coord[4]:coord[5], coord[2]:coord[3], coord[0]:coord[1]]
        else:
            image = image[coord[4]:coord[5], coord[2]:coord[3], coord[0]:coord[1]]
            image = image[None,:,:,:]

        return image

    def roi_coord(self, mask, roi='lungs'):
        frame_list = []
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

        if roi == 'lungs':
            mask[mask == 2] = 1

        for i in range(len(mask)):
            if mask[i].max() > 0:
                frame_list.append(i)

                ct_slice = mask[i]
                ct_slice = ct_slice[None, :, :]

                bbx = masks_to_boxes(ct_slice)
                bbx = bbx[0].detach().tolist()

                if bbx[0] < x_min: x_min = int(bbx[0])
                if bbx[1] < y_min: y_min = int(bbx[1])
                if bbx[2] > x_max: x_max = int(bbx[2])
                if bbx[3] > y_max: y_max = int(bbx[3])

        z_min = frame_list[0]
        z_max = frame_list[-1]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def read_ct(self, path):
            reader = sitk.ImageSeriesReader()
            dcm_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dcm_names)
            image = reader.Execute()
            return image
    
    def save_img(self, patient, image, mode='full_ct', roi=None, is_Label=False):
        if not os.path.exists(os.path.join(self.dataoutput, mode, patient)):
            os.makedirs(os.path.join(self.dataoutput, mode, patient))
        if roi is None:
            if is_Label:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_seg.pt'))
            else:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_ct.pt'))

        else:
            if is_Label:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_' + roi + '_seg.pt'))
            else:
                torch.save(image, os.path.join(self.dataoutput, mode, patient, patient + '_' + roi + '_ct.pt'))

    def normalize(self, image, space=(1,1,1.5), out_shape=(256, 256, 256), is_label=False):
        if is_label:
            # image = tt.Resample(space, image_interpolation='nearest')(image)
            image = tt.Resize(out_shape, image_interpolation='nearest')(image)
            image = tt.RescaleIntensity(out_min_max=(0,1))(image)
        else:
            image = tt.Resample(space, image_interpolation='bspline')(image)
            image = tt.Resize(out_shape, image_interpolation='bspline')(image)
            image = tt.Clamp(out_min= -200, out_max=250)(image)
            image = tt.RescaleIntensity(out_min_max=(0,1))(image)
        return image

    def tumor_bbx(self, outpath):
        print('Generating tumor bbx', self.dataset)
        print(outpath)
    
        patients_list = sorted(os.listdir(outpath))
        for pat in patients_list:
            print(pat)
            element_list = sorted(os.listdir(os.path.join(outpath, pat)))
            for element in element_list:
                if "_seg" not in element: continue
                patient_seg = torch.load(os.path.join(outpath, pat, element))
                tumor_cord = self.roi_coord(torch.tensor(patient_seg[0])) # x_min, x_max, y_min, y_max, z_min, z_max

                mask = np.zeros(patient_seg.shape[1:])
                mask[tumor_cord[4]:tumor_cord[5]+1, tumor_cord[2]:tumor_cord[3]+1, tumor_cord[0]:tumor_cord[1]+1] = 1
                torch.save(mask[None,:,:,:], os.path.join(outpath, pat, pat + '_bbx.pt'))