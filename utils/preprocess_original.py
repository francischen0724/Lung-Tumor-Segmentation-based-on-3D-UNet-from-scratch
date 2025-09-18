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
import torchio as tio
from scipy import ndimage

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
        print('Preprocessing MSD dataset...')

        print(self.datapath)
        ct_list = sorted(os.listdir(os.path.join(self.datapath, 'imagesTr')))
        seg_list = sorted(os.listdir(os.path.join(self.datapath, 'labelsTr')))

        for pat, label in zip(ct_list, seg_list):
            patient = pat.split('.')[0]
            print(patient)

            ct = nib.load(os.path.join(self.datapath, 'imagesTr', pat))
            ct = ct.get_fdata()
            ct = np.transpose(ct, (2,1,0))
            ct = np.flip(ct, axis=1).copy()

            seg = nib.load(os.path.join(self.datapath, 'labelsTr', label))
            seg = seg.get_fdata()
            seg = np.transpose(seg, (2,1,0))
            seg = np.flip(seg, axis=1).copy()

            preprocessed_ct = self.normalize(ct[None, :, :, :])
            print('CT Normalized')
            self.save_img(patient, preprocessed_ct)
            
            preprocessed_seg = self.normalize(seg[None, :, :, :], is_label=True)
            print('Seg Normalized')
            self.save_img(patient, preprocessed_seg, is_Label=True)

            self.extract_lungs(patient, ct, seg[None, :, :, :])

        self.tumor_bbx(self.dataoutput+'/lungs_roi')
        return


    def radiomics(self):
        print('Preprocessing Radiomics dataset...', flush=True)

        for patient in sorted(os.listdir(self.datapath)):
            patient_path = os.path.join(self.datapath, patient)
            if not os.path.isdir(patient_path):
                continue

            # 获取唯一 Study 文件夹
            study_dirs = [d for d in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, d))]
            if not study_dirs:
                print(f"❌ No study directory found in {patient}", flush=True)
                continue

            study_path = os.path.join(patient_path, study_dirs[0])
            print(f"📂 Patient: {patient}, Study folder: {study_dirs[0]}", flush=True)
            print(f"📁 Elements inside study: {os.listdir(study_path)}", flush=True)

            # ====== 读取 CT ======
            ct_path = os.path.join(study_path, "CT")
            if os.path.exists(ct_path):
                print(f"📥 Reading CT from: {ct_path}", flush=True)
                ct_sitk = self.read_ct(ct_path)
                ct_np = sitk.GetArrayFromImage(ct_sitk)
                pre_ct = self.normalize(ct_np[None, :, :, :])
                self.save_img(patient, pre_ct)
                patient_ct = ct_np
            else:
                print(f"❌ CT folder not found for {patient}", flush=True)
                continue

            # ====== 读取 SEG ======
            seg_np = None
            seg_path = os.path.join(study_path, "SEG")
            if os.path.exists(seg_path):
                dcm_files = [f for f in os.listdir(seg_path) if f.endswith(".dcm")]
                if not dcm_files:
                    print(f"❌ No DICOM file in seg folder for {patient}", flush=True)
                else:
                    seg_dcm_path = os.path.join(seg_path, dcm_files[0])
                    seg_dcm = dicom.read_file(seg_dcm_path)
                    seg_array = seg_dcm.pixel_array
                    seg_num = len(seg_dcm.SegmentSequence)
                    print(f"🧠 Seg shape: {seg_array.shape}, {seg_num} segments", flush=True)

                    seg_idx = 0
                    for i in range(seg_num):
                        label = seg_dcm.SegmentSequence[i].SegmentLabel
                        if label == "Neoplasm, Primary":
                            print(f"✅ Found Primary Neoplasm at slice {i}", flush=True)
                            seg_idx = i
                            break

                    dim0 = int(seg_array.shape[0] / seg_num)
                    seg_tensor = torch.reshape(
                        torch.from_numpy(seg_array),
                        (seg_num, dim0, 512, 512)
                    )
                    seg_np = seg_tensor[seg_idx].unsqueeze(0).numpy()
                    pre_seg = self.normalize(seg_np, is_label=True)
                    self.save_img(patient, pre_seg, is_Label=True)
            else:
                print(f"⚠️ No seg folder for {patient}, skipping segmentation.", flush=True)

            # ====== ROI ======
            if seg_np is not None:
                self.extract_lungs(patient, patient_ct, seg_np)
            else:
                self.extract_lungs(patient, patient_ct)
            

        self.tumor_bbx(self.dataoutput + '/lungs_roi')






    # def radiogenomics(self):

    #     print('Preprocessing Radiogenomics dataset...')

    #     patients_list =  sorted(os.listdir(self.datapath))
    #     for patient in patients_list:
    #         print(patient)

    #         patient_path = os.path.join(self.datapath, patient, 'CT')
    #         patient_ct = self.read_ct(patient_path)
    #         patient_ct = sitk.GetArrayFromImage(patient_ct)
    #         preprocessed_ct = self.normalize(patient_ct[None, :, :, :])
    #         self.save_img(patient, preprocessed_ct)

    #         if os.path.exists(os.path.join(self.datapath, patient, 'seg')):
    #             patient_seg = self.read_ct(os.path.join(self.datapath, patient, 'seg'))
    #             patient_seg = sitk.GetArrayFromImage(patient_seg)
                
    #             preprocessed_seg = self.normalize(patient_seg, is_label=True)
    #             self.save_img(patient, preprocessed_seg, is_Label=True)

    #             self.extract_lungs(patient, patient_ct, patient_seg)
    #             self.extract_tumor(patient, patient_ct, torch.tensor(patient_seg[0]))
            
    #         else:
    #             self.extract_lungs(patient, patient_ct)
    #             continue
    #     self.tumor_bbx(self.dataoutput+'/lungs_roi')

    def radiogenomics(self):
        print('Preprocessing Radiogenomics dataset...', flush=True)

        # for patient in sorted(os.listdir(self.datapath)):
        #     patient_path = os.path.join(self.datapath, patient)
        #     if not os.path.isdir(patient_path):
        #         continue

        # 遍历患者文件夹，从第22个（index=21）开始处理
        for i, patient in enumerate(sorted(os.listdir(self.datapath))):
            if i < 22:
                continue  # 跳过前21个病人（第23个是出错的那个）

            patient_path = os.path.join(self.datapath, patient)
            if not os.path.isdir(patient_path):
                continue
        
            # 查找带有 -CT 的study目录（跳过PET等）
            study_dirs = [d for d in os.listdir(patient_path)
                        if os.path.isdir(os.path.join(patient_path, d)) and '-CT' in d.upper()]
            if not study_dirs:
                print(f"❌ No -CT directory found in {patient}", flush=True)
                continue

            study_path = os.path.join(patient_path, study_dirs[0])
            print(f"📂 Patient: {patient}, CT Study folder: {study_dirs[0]}", flush=True)
            print(f"📁 Elements inside study: {os.listdir(study_path)}", flush=True)

            ct_folder = None
            seg_folder = None

            # 遍历 study 目录下所有子文件夹，区分 CT 与 segmentation
            for sub in os.listdir(study_path):
                sub_path = os.path.join(study_path, sub)
                if not os.path.isdir(sub_path):
                    continue
                if 'segmentation' in sub.lower():
                    seg_folder = sub_path
                else:
                    ct_folder = sub_path  # 默认另一个为 CT

            if ct_folder is None:
                print(f"❌ CT folder not found for {patient}", flush=True)
                continue

            # ====== 读取 CT ======
            print(f"📥 Reading CT from: {ct_folder}", flush=True)
            try:
                ct_sitk = self.read_ct(ct_folder)
                ct_np = sitk.GetArrayFromImage(ct_sitk)
                pre_ct = self.normalize(ct_np[None, :, :, :])
                self.save_img(patient, pre_ct)
                patient_ct = ct_np
            except Exception as e:
                print(f"❌ Error reading CT for {patient}: {e}", flush=True)
                continue

            # ====== 读取 SEG ======
            seg_np = None
            if seg_folder and os.path.exists(seg_folder):
                dcm_files = [f for f in os.listdir(seg_folder) if f.endswith(".dcm")]
                if not dcm_files:
                    print(f"❌ No DICOM file in seg folder for {patient}", flush=True)
                else:
                    seg_dcm_path = os.path.join(seg_folder, dcm_files[0])
                    try:
                        import pydicom
                        seg_dcm = pydicom.dcmread(seg_dcm_path)
                        seg_array = seg_dcm.pixel_array
                        seg_num = len(seg_dcm.SegmentSequence)
                        print(f"🧠 Seg shape: {seg_array.shape}, {seg_num} segments", flush=True)

                        seg_idx = 0
                        for i in range(seg_num):
                            label = seg_dcm.SegmentSequence[i].SegmentLabel
                            if label == "Neoplasm, Primary":
                                print(f"✅ Found Primary Neoplasm at slice {i}", flush=True)
                                seg_idx = i
                                break

                        dim0 = int(seg_array.shape[0] / seg_num)
                        seg_tensor = torch.reshape(
                            torch.from_numpy(seg_array),
                            (seg_num, dim0, 512, 512)
                        )
                        seg_np = seg_tensor[seg_idx].unsqueeze(0).numpy()
                        pre_seg = self.normalize(seg_np, is_label=True)
                        self.save_img(patient, pre_seg, is_Label=True)
                    except Exception as e:
                        print(f"❌ Error processing segmentation for {patient}: {e}", flush=True)

            else:
                print(f"⚠️ No segmentation folder for {patient}", flush=True)

            # ====== ROI ======
            if seg_np is not None:
                try:
                    self.extract_lungs(patient, patient_ct, seg_np)
                    self.extract_tumor(patient, patient_ct, torch.tensor(seg_np[0]))
                except Exception as e:
                    print(f"⚠️ Skipping {patient} due to lung/tumor extraction error: {e}", flush=True)
                    continue
            else:
                try:
                    self.extract_lungs(patient, patient_ct)
                except Exception as e:
                    print(f"⚠️ Skipping {patient} due to lung extraction error (no seg): {e}", flush=True)
                    continue

        self.tumor_bbx(self.dataoutput + '/lungs_roi')


    def extract_tumor(self, patient, ct, seg=None):
        tcr = self.roi_coord(seg, roi='tumor') # tcr = tumor_roi_coord
        tumor_ct = ct.copy()
        tumor_ct = self.crop_coord(tcr, tumor_ct)
        tumor_ct = self.normalize(tumor_ct, out_shape=(64, 64, 64))
        self.save_img(patient, tumor_ct, mode='tumor_roi')

    def extract_lungs(self, patient, ct, seg=None):

        model_path = os.path.join(os.path.dirname(__file__), 'R231.pth')
        model = mask.get_model('unet', modelpath=model_path).to(self.device)
        extracted = mask.apply(ct, model)
        extracted = torch.tensor(extracted)

        lungs_mask = extracted.clone()
        lcr = self.roi_coord(lungs_mask, roi='lungs') # lcr = lung_roi_coord
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

    # def normalize(self, image, space=(1,1,1.5), out_shape=(256, 256, 256), is_label=False):
    #     if is_label:
    #         # image = tt.Resample(space, image_interpolation='nearest')(image)
    #         image = tt.Resize(out_shape, image_interpolation='nearest')(image)
    #         image = tt.RescaleIntensity(out_min_max=(0,1))(image)
    #     else:
    #         image = tt.Resample(space, image_interpolation='bspline')(image)
    #         image = tt.Resize(out_shape, image_interpolation='bspline')(image)
    #         image = tt.Clamp(out_min= -200, out_max=250)(image)
    #         image = tt.RescaleIntensity(out_min_max=(0,1))(image)
    #     return image

    def normalize(self, image, space=(1,1,1.5), out_shape=(256, 256, 256), is_label=False):
        if is_label:
            image = tt.Resample(space, image_interpolation='nearest')(image)
            image = tt.Resize(out_shape, image_interpolation='nearest')(image)
            # ⚠️ 不要再对标签图像进行 intensity scaling
        else:
            image = tt.Resample(space, image_interpolation='bspline')(image)
            image = tt.Resize(out_shape, image_interpolation='bspline')(image)
            image = tt.Clamp(out_min=-200, out_max=250)(image)
            image = tt.RescaleIntensity(out_min_max=(0,1))(image)
        return image
    
    # # spacing problem
    # def normalize(self, image, space=(1,1,1.5), out_shape=(256, 256, 256), is_label=False):

    #     def spacing_from_affine(affine):
    #         return np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])

    #     def resample_numpy(image_np, original_spacing, target_spacing):
    #         if image_np.ndim != 3 or np.any(np.array(image_np.shape) == 0):
    #             raise ValueError(f"[resample_numpy] Invalid shape: {image_np.shape}")
    #         resize_factor = np.array(original_spacing) / np.array(target_spacing)
    #         new_real_shape = np.array(image_np.shape) * resize_factor
    #         real_resize_factor = new_real_shape / np.array(image_np.shape)
    #         if np.any(np.isnan(real_resize_factor)) or np.any(np.isinf(real_resize_factor)):
    #             raise ValueError(f"[resample_numpy] Invalid resize factor: {real_resize_factor}")
    #         return ndimage.zoom(image_np, real_resize_factor, order=0 if is_label else 3)

    #     # 兼容 numpy 直接传入的情况
    #     if isinstance(image, np.ndarray):
    #         tensor = torch.from_numpy(image)
    #         if tensor.ndim == 3:
    #             tensor = tensor.unsqueeze(0)  # (1, D, H, W)
    #         image_type = tio.LabelMap if is_label else tio.ScalarImage
    #         image = image_type(tensor=tensor, affine=np.eye(4))

    #     try:
    #         interp = 'nearest' if is_label else 'bspline'
    #         image = tio.Resample(space, image_interpolation=interp)(image)
    #         image = tio.Resize(out_shape, image_interpolation=interp)(image)

    #         if not is_label:
    #             image = tio.Clamp(out_min=-200, out_max=250)(image)
    #             image = tio.RescaleIntensity(out_min_max=(0,1))(image)

    #         return image

    #     except ValueError as e:
    #         if 'Spacing must be strictly positive' not in str(e):
    #             raise e

    #         print('[normalize] ⚠️ Invalid spacing detected, falling back to ndimage.zoom')

    #         # fallback: 重采样 + resize
    #         tensor = image.tensor.numpy()[0]
    #         spacing = spacing_from_affine(image.affine)
    #         spacing = [s if s > 0 else 1.0 for s in spacing]  # 替换 0 spacing

    #         # Step 1: spacing 重采样
    #         tensor = resample_numpy(tensor, spacing, space)

    #         # Step 2: resize 到目标形状
    #         resize_factor = np.array(tensor.shape) / np.array(out_shape)
    #         tensor = ndimage.zoom(tensor, 1 / resize_factor, order=0 if is_label else 3)

    #         # Step 3: 图像强度调整
    #         if not is_label:
    #             tensor = np.clip(tensor, -200, 250)
    #             tensor = (tensor + 200) / 450.0  # scale to 0~1

    #         tensor = torch.tensor(tensor)[None]  # (1, D, H, W)
    #         image_type = tio.LabelMap if is_label else tio.ScalarImage
    #         return image_type(tensor=tensor, affine=image.affine)





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