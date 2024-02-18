# import os
# import numpy as np
# import pandas as pd
# from config.config import config_params

# for key, value in config_params.items():
#     if isinstance(value, str):
#         exec(f"{key} = '{value}'")
#     else:
#         exec(f"{key} = {value}")
        
# df = pd.read_csv(f"{data_dir}/train_labels.csv")
# df = df.drop_duplicates()
# fold = 4
# train_df = df[(df['fold_patient'] != fold)] 
# valid_df = df[df['fold_patient'] == fold]
# test_df = pd.read_csv(f"{data_dir}/test_labels.csv")
# all_df = pd.read_csv(f"{data_dir}/labels.csv")
# all_df = all_df.drop_duplicates()
# # print(len(np.unique(df['patient_id'].tolist())))
# num_train, num_val, num_test, num_all = [], [], [], []

# for i in np.unique(all_df['patient_id'].tolist()):
#     num_all.append(len(os.listdir(f"{data_dir}/{i}/images")))
#     # print(os.listdir(f"{data_dir}/{i}/images/"))
    
# for i in np.unique(train_df['patient_id'].tolist()):
#     num_train.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
# for i in np.unique(valid_df['patient_id'].tolist()):
#     num_val.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
# for i in np.unique(test_df['patient_id'].tolist()):
#     num_test.append(len(os.listdir(f"{data_dir}/{i}/images")))

# # print(num_train, num_val, num_test, num_all, num_train + num_val + num_test)
# print("Comparison of number of slices among train, val and test data")
# print(f"All mean: {np.mean(num_all)} SD: {np.std(num_all)}")
# print(f"Train mean: {np.mean(num_train)} SD: {np.std(num_train)}")
# print(f"Val mean: {np.mean(num_val)} SD: {np.std(num_val)}")
# print(f"Test mean: {np.mean(num_test)} SD: {np.std(num_test)}")

# print(len(all_df))
# print(len(train_df) + len(valid_df) + len(test_df))
# print(len(np.unique(test_df['patient_id'].tolist())))
# print(np.unique(test_df['patient_id'].tolist()))

from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config.config import *
from utils import *
from tqdm import tqdm as T
import numpy as np
import cv2
import nrrd
import pandas as pd
# from p_tqdm import p_map
num_slices = 0
data_dir = "../DATA/lymph_node/ct_221"
new_data_dir = f"../DATA/lymph_node/ct_221_{num_slices}_npz"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
patient_ids = os.listdir(data_dir)

# for pat_id in T(patient_ids):
def data_processing(args):
    pat_id, num_slices = args
    pat_id = str(pat_id)
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'IM00' in f][0]
    except:
        print(f"Problem with {pat_id}")    
    seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation' in f][0]

    # img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
    mask_pat_id, metadata = nrrd.read(os.path.join(data_dir, pat_id, seg_file))
    # print(metadata)
    # padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
    # os.makedirs(os.path.join(new_data_dir, pat_id, 'images'), exist_ok=True)
    # os.makedirs(os.path.join(new_data_dir, pat_id, 'masks'), exist_ok=True)
    ln_lens = []
    len_ln = 0
    for i in range(mask_pat_id.shape[-1]):
        # img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], 40, 400, 0, 1)
        mask = mask_pat_id[:, :, i]
        mask[mask>0] = 255   
        # label_dict['patient_id'].append(pat_id)
        # label_dict['slice_num'].append(i)
        if np.sum(mask) != 0:
            len_ln += 1
        else:
            if len_ln !=0: ln_lens.append(len_ln)
            len_ln = 0
            
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.png'), img)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.png'), mask)
    if pat_id == '4793':
        print(ln_lens)
        print(f"Patient ID {pat_id} Mean {np.mean(ln_lens)} SD {np.std(ln_lens)} Max {np.max(ln_lens)} Min {np.min(ln_lens)}")
    return ln_lens

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

if __name__ == '__main__':
    args_list = [(patient_id, num_slices) for patient_id in patient_ids]
    LN_LENS = []
    for arg in T(args_list):
        LN_LENS.extend(data_processing(arg))
        # break
    print(f"Mean {np.mean(LN_LENS)} SD {np.std(LN_LENS)} Max {np.max(LN_LENS)} Min {np.min(LN_LENS)}")