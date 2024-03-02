from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch import inf
from config.config import *
from utils import *
from tqdm import tqdm as T
import numpy as np
import cv2
import nrrd
import pandas as pd
import pickle


# from p_tqdm import p_map
num_slices = 0
data_dir = "./data/lymph_node/ct_221"
new_data_dir = f"./data/lymph_node/ct_221_{num_slices}_slice"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
labels = {}
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

    img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
    mask_pat_id = nrrd.read(os.path.join(data_dir, pat_id, seg_file))[0]
    padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
    os.makedirs(os.path.join(new_data_dir, pat_id, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, pat_id, 'raw_slice'), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, pat_id, 'masks'), exist_ok=True)
    for i in range(num_slices, padded_data.shape[-1] - num_slices):
        img_8_bit, img_norm = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], 40, 400, 0, 1, rescale=True)
        mask = mask_pat_id[:, :, i - num_slices]
        mask[mask>0] = 255
        mask = mask.astype(np.uint8)
        label_dict['patient_id'].append(pat_id)
        label_dict['slice_num'].append(i)
        if np.sum(mask) == 0:
            label_dict['label'].append(0)
            labels[i] = 0
        else: 
            label_dict['label'].append(1)
            labels[i] = 1
        # np.savez(os.path.join(new_data_dir, pat_id, f'images/{i-num_slices}'), img)
        # np.savez(os.path.join(new_data_dir, pat_id, f'masks/{i-num_slices}'), mask)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.png'), img)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.png'), mask)
        cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.jpg'), img_8_bit)
        cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.jpg'), mask)
        np.save(os.path.join(new_data_dir, pat_id, f'raw_slice/{i}.npy'), img_norm)
    
    save_label_path = os.path.join(new_data_dir, pat_id, 'labels.pkl')
    with open(save_label_path, 'wb') as f:
        pickle.dump(labels, f)

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

if __name__ == '__main__':
    
    args_list = [(patient_id, num_slices) for patient_id in patient_ids]
    with Pool(16) as p:
        list(T(p.imap(data_processing, args_list), total=len(patient_ids), colour='red'))
    
    # df = pd.DataFrame(label_dict)
    # df.to_csv(f"{new_data_dir}/labels.csv", index=False)
    # gkf = GroupKFold(n_splits=5)
    # patient_ids = df.patient_id.unique().tolist()
    # patient_ids.sort()
    # train_pat_ids = patient_ids[:160]
    # test_pat_ids = patient_ids[160:]
    # df['path'] = list(map(datapath, df['patient_id'], df['slice_num']))   
    # df['path'] = df['path'].map(lambda x: f"{new_data_dir}/{x}")
    # train_df = df[df["patient_id"].isin(train_pat_ids)].reset_index(drop=True)
    # test_df = df[df["patient_id"].isin(test_pat_ids)].reset_index(drop=True)

    # # for i, (train_index, val_index) in enumerate(gkf.split(train_df['path'], train_df['label'], groups=train_df['patient_id'])):
    # #     train_idx = train_index
    # #     val_idx = val_index
    # #     train_df.loc[val_idx, 'fold'] = i

    # pat_id_dict = {}

    # for i, pat_id in enumerate(train_pat_ids):
    #     pat_id_dict.update({pat_id:i*5//len(train_pat_ids)})

    # train_df['fold_patient'] = train_df['patient_id'].map(lambda x: pat_id_dict[x])

    # # train_df['fold'] = train_df['fold'].astype('int')
    # train_df['fold_patient'] = train_df['fold_patient'].astype('int')
    # # df['fold'] = df['patient_id'].map(lambda x: 0 if x in train_pat_ids else 1)
    # print(train_df.head(20))

    # train_df.to_csv(f"{new_data_dir}/train_labels.csv", index=False)
    # test_df.to_csv(f"{new_data_dir}/test_labels.csv", index=False)