from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config import *
from utils import *
from tqdm import tqdm as T
import numpy as np
import cv2
import nrrd
import pandas as pd
# from p_tqdm import p_map
data_dir = "../DATA/lymph_node/ct_221"
new_data_dir = "../DATA/lymph_node/ct_221_npz"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
patient_ids = os.listdir(data_dir)

# for pat_id in T(patient_ids):
def data_processing(pat_id):
    pat_id = str(pat_id)
    try:
        data_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'IM00' in f][0]
    except:
        print(f"Problem with {pat_id}")    
    seg_file = [f for f in os.listdir(os.path.join(data_dir, pat_id))if 'Segmentation' in f][0]

    img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
    mask_pat_id = nrrd.read(os.path.join(data_dir, pat_id, seg_file))[0]
    os.makedirs(os.path.join(new_data_dir, pat_id, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_data_dir, pat_id, 'masks'), exist_ok=True)
    for i in range(img_pat_id.shape[-1]):
        img = window_image(img_pat_id[:,:,i], 40, 400, 0, 1)
        mask = mask_pat_id[:, :, i]
        mask[mask>0] = 255
        label_dict['patient_id'].append(pat_id)
        label_dict['slice_num'].append(i)
        if np.sum(mask) == 0:
            label_dict['label'].append(0)
        else: label_dict['label'].append(1)
        np.savez(os.path.join(new_data_dir, pat_id, f'images/{i}'), img)
        np.savez(os.path.join(new_data_dir, pat_id, f'masks/{i}'), mask)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.png'), img)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.png'), mask)

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

if __name__ == '__main__':
    with Pool(16) as p:
        list(T(p.imap(data_processing, patient_ids), total=len(patient_ids), colour='red'))
    df = pd.DataFrame(label_dict)
    df.to_csv(f"{new_data_dir}/labels.csv", index=False)
    gkf = GroupKFold(n_splits=5)
    patient_ids = df.patient_id.unique().tolist()
    patient_ids.sort()
    train_pat_ids = patient_ids[:160]
    test_pat_ids = patient_ids[160:]
    df['path'] = list(map(datapath, df['patient_id'], df['slice_num']))   
    df['path'] = df['path'].map(lambda x: f"{new_data_dir}/{x}")
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

    train_df.to_csv(f"{new_data_dir}/train_labels.csv", index=False)
    test_df.to_csv(f"{new_data_dir}/test_labels.csv", index=False)