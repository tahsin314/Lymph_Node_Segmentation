import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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


num_slices = 2
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
data_dir = 'data'
new_data_dir = f"{data_dir}_slices_{num_slices}_npz"
check_data_dir = f"{data_dir}_check"
patient_ids = os.listdir(data_dir)



def window_image(img, window_center=940, window_width=2120, 
intercept=0, slope=1, rescale=True):
    # for thyroid window range is -200 to 2000 
    # transform to hu
    img = (img*slope +intercept) #for translation adjustments given in the dicom file. 
    img_min = window_center - window_width//2 #minimum HU level
    img_max = window_center + window_width//2 #maximum HU level
    img[img<img_min] = img_min #set img_min for all HU levels less than minimum HU level
    img[img>img_max] = img_max #set img_max for all HU levels higher than maximum HU level
    if rescale: 
        img = ((img - img_min) / (img_max - img_min)*255.0).astype('uint8') 
    return img


def convert_save_segmentation_mask(pat_id):

    data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if '.nrrd' in f and 'Segmentation' not in f][0]
    seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation.seg' in f][0]
    
    os.makedirs(os.path.join(check_data_dir, pat_id, 'images'), exist_ok=True)
    os.makedirs(os.path.join(check_data_dir, pat_id, 'masks'), exist_ok=True)

    image_path = os.path.join(data_dir, pat_id, data_file)
    img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
    
    segmentation_file_path = os.path.join(data_dir, pat_id, seg_file)
    mask_array, header = nrrd.read(segmentation_file_path)
    mask_array = np.transpose(mask_array, axes=[2, 1, 0])
    
    image = sitk.ReadImage(image_path, imageIO="NrrdImageIO")
    image_array = sitk.GetArrayFromImage(image)
    lbl_array = np.zeros_like(image_array, dtype=np.uint8)
    
    #converting segmentation mask to original image size
    offset = header['Segmentation_ReferenceImageExtentOffset'].split()
    offset_width, offset_height, offset_depth = [int(value) for value in offset]
    #print(offset_width, offset_height, offset_depth)
    mask_depth, mask_height, mask_width = mask_array.shape
    depth_slice = slice(offset_depth, offset_depth + mask_depth)
    height_slice = slice(offset_height, offset_height + mask_height)
    width_slice = slice(offset_width, offset_width + mask_width)
    lbl_array[depth_slice, height_slice, width_slice] = mask_array
    
    padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
    print(padded_data.shape)
    for i in range(num_slices, padded_data.shape[-1] - num_slices):
        #print('Checking for ground truth label,array sum: ', np.sum(lbl_array[i]))
        img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1])
        mask = lbl_array[i-num_slices]
        mask[mask>0] = 255
        
        label_dict['patient_id'].append(pat_id)
        label_dict['slice_num'].append(i)
        if np.sum(mask) == 0:
            label_dict['label'].append(0)
        else: label_dict['label'].append(1)
        
        np.savez(os.path.join(check_data_dir, pat_id, f'images/{i-num_slices}'), img)
        np.savez(os.path.join(check_data_dir, pat_id, f'masks/{i-num_slices}'), mask)
        
        cv2.imwrite(os.path.join(check_data_dir, pat_id, f'masks/{i-num_slices}.png'), mask)
        for j in range(img.shape[-1]):
            path = os.path.join(os.path.join(check_data_dir, pat_id, 'images', str(i)))
            os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, f'{j}.png'), img[:,:,j])



        # img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], 40, 400, 0, 1)
        # img = window_image(img_pat_id[:,:,i])
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # img = cv2.flip(img, 1)
        # cv2.imwrite(os.path.join(check_data_dir, pat_id, f'images/{i}.png'), img)
        #cv2.imwrite(os.path.join(path, str(i)+'.png'), lbl_array[i]*255)
    message = 'Masks img folder done'
    return message

#block for only :main images
#for pat_id in patient_ids:
if __name__ == '__main__':
    args_list = [(patient_id) for patient_id in patient_ids]
    with Pool(16) as p:
        list(T(p.imap(convert_save_segmentation_mask, args_list), total=len(patient_ids), colour='red'))


