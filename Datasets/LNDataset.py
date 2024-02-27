import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import gc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from platform import system
import os
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS']=str(2**64)
# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import platform
import warnings
warnings.filterwarnings('ignore')

class LNDataset(Dataset):
    def __init__(self, image_ids, labels=None, dim=256, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.dim = dim
        self.transforms = transforms

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_id = image_id.replace('ct_221_8bit', 'ct_221_npz')
        # image = cv2.imread(image_id)
        image = np.load(image_id.replace('png', 'npz'), allow_pickle=True)['arr_0']
        image = cv2.resize(image, (self.dim, self.dim))
        # num_chans = image.shape[-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
        
        mask = np.load(image_id.replace('/images/', '/masks/').replace('png', 'npz'), allow_pickle=True)['arr_0']
        mask = cv2.resize(mask, (self.dim, self.dim)) #//255.
        # print(np.max(image), np.max(mask))
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)//255.     
        if self.transforms is not None:
            image, mask = self.transforms.generation(image, mask)
        else:
            image = image.reshape(self.dim, self.dim, -1).transpose(2, 0, 1)
            mask = mask.reshape(self.dim, self.dim)
            mask = (mask > 0.5).astype(np.uint8)
            one_hot_mask = np.zeros((2, self.dim, self.dim), dtype=np.uint8)
            one_hot_mask[0, :, :] = (mask == 0).astype(np.uint8)
            one_hot_mask[1, :, :] = (mask == 1).astype(np.uint8)
            # if np.sum(mask)>0:
            #     print(np.argmax(one_hot_mask, axis=0).shape)
            #     cv2.imwrite(f"random_masks/{image_id.split('/')[-1]}", 255*np.argmax(one_hot_mask, axis=0))
        # print(np.min(image), np.max(image), np.min(mask), np.max(mask))
        return image, mask

    def __len__(self):
        return len(self.image_ids)
    
    def onehot(self, num_class, target):
        vec = torch.zeros(num_class, dtype=torch.float32)
        vec[target.astype('int')] = 1.
        return vec
    
    def get_labels(self):
        return list(self.labels)

        