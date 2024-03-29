import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
from tqdm import tqdm as T
import nrrd
import copy
import scipy
from PIL import Image
import SimpleITK as sitk
import shutil #extra functions for working with files
import matplotlib.pyplot as plt
from multiprocessing import Pool

from utils import window_image


num_slices = 2
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}

# path in server
data_dir = '/shared/rail_lab/thyroid_cartilage/data'
root_dir = '/shared/rail_lab/thyroid_cartilage/'

dir_name = f'thyroid_slices_{num_slices}_npz'
check_data_dir = os.path.join(root_dir, dir_name)
os.makedirs(check_data_dir, exist_ok=True)

img_dir_name = f'thyroid_slices_{num_slices}_jpg'
check_image_dir = os.path.join(root_dir, img_dir_name)
os.makedirs(check_image_dir, exist_ok=True)

patient_ids = os.listdir(data_dir)

def convert_save_segmentation_mask(pat_id):
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id)) if ('.nrrd' in f) and ('Segmentation' not in f) and ('Image' not in f)][0]
    except:
        print(f'pat_id:{pat_id} data_file doesnt exist')
        return

    try:
        seg_file = [j for j in os.listdir(os.path.join(data_dir, pat_id)) if ('Segmentation.seg' in j) or ('Image.nrrd' in j)][0]
    except:
        print(f'pat_id:{pat_id} seg_file doesnt exist') 
        return
    
    os.makedirs(os.path.join(check_data_dir, pat_id, 'images'), exist_ok=True)
    os.makedirs(os.path.join(check_data_dir, pat_id, 'masks'), exist_ok=True)

    image_path = os.path.join(data_dir, pat_id, data_file)
    img_pat_id = nrrd.read(image_path)[0]
    
    segmentation_file_path = os.path.join(data_dir, pat_id, seg_file)
    mask_array, header = nrrd.read(segmentation_file_path)
    mask_array = np.transpose(mask_array, axes=[2, 1, 0])
    
    image = sitk.ReadImage(image_path, imageIO="NrrdImageIO")
    image_array = sitk.GetArrayFromImage(image)
    lbl_array = np.zeros_like(image_array, dtype=np.uint8)
    
    #converting segmentation mask to original image size
    offset = header['Segmentation_ReferenceImageExtentOffset'].split()
    offset_width, offset_height, offset_depth = [int(value) for value in offset]
    mask_depth, mask_height, mask_width = mask_array.shape

    if offset_depth + mask_depth > lbl_array.shape[0]:
        diff = offset_depth + mask_depth - lbl_array.shape[0]
        print(f'pat id:{pat_id} mask_Depth:{mask_depth} offset_depth:{offset_depth} lbl_array:{lbl_array.shape} diff:{diff}')
    else:
        diff = 0
        
    depth_slice = slice(offset_depth, offset_depth + mask_depth - diff)
    height_slice = slice(offset_height, offset_height + mask_height)
    width_slice = slice(offset_width, offset_width + mask_width)
    lbl_array[depth_slice, height_slice, width_slice] = mask_array[:mask_depth-diff, :, :]
    
    padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
    

    for i in range(num_slices, padded_data.shape[-1] - num_slices):
        #print('Checking for ground truth label,array sum: ', np.sum(lbl_array[i]))
        img_8_bit, img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], window_center=940, window_width=2120, intercept=0, slope=1, rescale=True)
        mask = lbl_array[i-num_slices]
        mask[mask>0] = 255
        
        label_dict['patient_id'].append(pat_id)
        label_dict['slice_num'].append(i-num_slices)
        if np.sum(mask) == 0:
            label_dict['label'].append(0)
        else: label_dict['label'].append(1)
        
        rotated = np.rot90(img, k=-1, axes=(0,1))
        img = np.flip(rotated, axis=1)

        os.makedirs(os.path.join(check_data_dir, pat_id, 'images'), exist_ok=True)
        os.makedirs(os.path.join(check_data_dir, pat_id, 'masks'), exist_ok=True)
        np.savez(os.path.join(check_data_dir, pat_id, f'images/{i-num_slices}'), img)
        np.savez(os.path.join(check_data_dir, pat_id, f'masks/{i-num_slices}'), mask)

        # mask_img_path = os.path.join(check_image_dir, pat_id, 'masks')
        # img_path = os.path.join(check_image_dir, pat_id, 'images')
        # os.makedirs(mask_img_path, exist_ok=True)
        # os.makedirs(img_path, exist_ok=True)
        # # writing images and masks
        # cv2.imwrite(os.path.join(mask_img_path, f'{i-num_slices}.png'), mask)
        # if i==num_slices:
        #     for j in range(img_8_bit.shape[-1]):
        #         cv2.imwrite(os.path.join(img_path, f'{j}.png'), flipped[:,:,j])
        # else:
        #     cv2.imwrite(os.path.join(img_path, f'{i+num_slices}.png'), flipped[:,:,-1])
    message = 'Masks img folder done'
    return label_dict

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

#block for only :main images
#for pat_id in patient_ids:
if __name__ == '__main__':
    args_list = [(patient_id) for patient_id in patient_ids]
    with Pool(32) as p:
        results = list(T(p.imap(convert_save_segmentation_mask, args_list), total=len(patient_ids), colour='red'))
     # Merge results iteratively using a loop:
    merged_label_dict = {}

    for d in results:
        for key, value in d.items():
            if key in merged_label_dict:
                merged_label_dict[key].extend(value)
            else:
                merged_label_dict[key] = value

    df = pd.DataFrame(merged_label_dict).drop_duplicates()
    df.to_csv(f"{check_data_dir}/labels.csv", index=False)
    gkf = GroupKFold(n_splits=5)
    patient_ids = df.explode('patient_id')['patient_id'].unique().tolist()
    patient_ids.sort()
    train_pat_ids = patient_ids[:300]
    test_pat_ids = patient_ids[300:]
    df['path'] = list(map(datapath, df['patient_id'], df['slice_num']))   
    df['path'] = df['path'].map(lambda x: f"{check_data_dir}/{x}")
    train_df = df[df["patient_id"].isin(train_pat_ids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pat_ids)].reset_index(drop=True)

    # for i, (train_index, val_index) in enumerate(gkf.split(train_df['path'], train_df['label'], groups=train_df['patient_id'])):
    #     train_idx = train_index
    #     val_idx = val_index
    #     train_df.loc[val_idx, 'fold'] = i

    pat_id_dict = {}

    for i, pat_id in enumerate(train_pat_ids):
        pat_id_dict.update({pat_id:i*5//len(train_pat_ids)})

    train_df['fold_patient'] = train_df['patient_id'].map(lambda x: pat_id_dict[x])

    # train_df['fold'] = train_df['fold'].astype('int')
    train_df['fold_patient'] = train_df['fold_patient'].astype('int')
    # df['fold'] = df['patient_id'].map(lambda x: 0 if x in train_pat_ids else 1)
    print(train_df.head(20))

    train_df.to_csv(f"{check_data_dir}/train_labels.csv", index=False)
    test_df.to_csv(f"{check_data_dir}/test_labels.csv", index=False)