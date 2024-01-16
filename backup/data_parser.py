import os
import numpy as np
import natsort
import random
import pickle

def file_save(data, save_path, file_name):
    if not os.path.exists(os.path.join(save_path, file_name)):
        with open(os.path.join(save_path, file_name), 'wb') as f:
            pickle.dump(data, f)


def create_split(patient_ids, data_path, save_path, train=False, val=False):
    # based on patient id/ patient number
    if train:
        #160 patients --> 23k slices
        # 9k slices ---> containns lymph nodes --> pos slices
        # 14k slices don't have lymph nodes --> neg slices
        split_ids = patient_ids[:160]
        folder = "train"
        
    elif val:
        split_ids = patient_ids[160:200]
        folder = "val"
    else:
        split_ids = patient_ids[200:]
        folder = "test"

    pos_sample_file = "lymph_node.pkl"
    neg_sample_file = "no_lymph_node.pkl"

    lymph_node_data = []
    no_lymph_node_data = []
    for pat_id in split_ids:
        #print('pat id: ',pat_id)
        data_folder = os.path.join(data_path, pat_id, 'raw_slice')
        label_dict = pickle.load(open(os.path.join(data_path, pat_id, 'labels.pkl'), 'rb'))
        slice_ids = os.listdir(data_folder)
        for i, img_id in enumerate(slice_ids):
            slice_path = os.path.join(data_folder, img_id)
            slice_num = int(img_id[:img_id.rindex('.')])
            label = label_dict[slice_num]       # each slice label --> 0/1
            if label==1:
                lymph_node_data.append(slice_path)
            else:
                no_lymph_node_data.append(slice_path)
    save_path = os.path.join(save_path, folder)
    os.makedirs(save_path, exist_ok=True)
    print(f'{folder}::LN slices:{len(lymph_node_data)} blank:{len(no_lymph_node_data)}')
    file_save(lymph_node_data, save_path, pos_sample_file)
    file_save(no_lymph_node_data, save_path, neg_sample_file)

if __name__=='__main__':
    data_path = "./data/lymph_node/ct_221_0_slice"
    save_path = "./data/lymph_node/split"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    patient_ids = os.listdir(data_path)
    patient_ids = natsort.natsorted(patient_ids)
    print('patient ids: ',patient_ids)
    #random.shuffle(patient_ids)
    create_split(patient_ids, data_path, save_path, train=True)
    create_split(patient_ids, data_path, save_path, val=True)
    create_split(patient_ids, data_path, save_path)







