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
import random
warnings.filterwarnings('ignore')

class LNDataset(Dataset):
    def __init__(self, image_ids, labels=None, dim=384, transforms=None):
        super().__init__()
        self.image_ids = image_ids
        self.labels = labels
        self.dim = dim
        self.transforms = transforms

        self.transform_val = A.Compose(
            [
                A.CenterCrop (height = dim, width = dim)
            ],
            additional_targets = {'image0':'image', 'image1':'image'}
        )
        if self.transforms is not None:
            random.shuffle(self.image_ids)

    def convert_to_tensor(self, img):
        img = np.expand_dims(img, axis=2)
        img_torch = torch.from_numpy(img)
        img_torch = img_torch.type(torch.FloatTensor)
        img_torch = img_torch.permute(-1, 0, 1)
        return img_torch

    def onehot(self, num_class, target):
        vec = torch.zeros(num_class, dtype=torch.float32)
        vec[target] = 1.
        return vec

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = np.load(image_id)
        img_8_bit_path = image_id.replace('/raw_slice/', '/images/').replace('.npy','.jpg')
        image_8_bit = cv2.imread(img_8_bit_path)
        num_chans = image_8_bit.shape[-1]   
        
        mask = cv2.imread(image_id.replace('/raw_slice/', '/masks/'))
    
        image_8_bit = image_8_bit[:,:,0]
        mask = mask[:,:,0]

        if self.transforms is not None:
            image, mask, image_8_bit = self.transforms.generation(image, mask, image_8_bit)
            #print('transformed')
        else:
            transformed = self.transform_val(image=image, image0=mask, image1=image_8_bit)
            image, mask, image_8_bit = transformed['image'], transformed['image0'], transformed['image1']
        

        image_8_bit = image_8_bit/255.0
        mask = mask/255
        mask = mask.astype('uint8')

        img_tensor = self.convert_to_tensor(image)
        img_8_bit_tensor = self.convert_to_tensor(image_8_bit)
        mask_tensor = self.convert_to_tensor(mask)

        mask_value = np.count_nonzero(mask)
        if mask_value>0:
            labels = 1
        else:
            labels = 0
        target = self.onehot(self.num_class, labels) 

        return img_tensor, img_8_bit_tensor, mask_tensor, target

    def __len__(self):
        return len(self.image_ids)
    
    def onehot(self, num_class, target):
        vec = torch.zeros(num_class, dtype=torch.float32)
        vec[target.astype('int')] = 1.
        return vec
    
    def get_labels(self):
        return list(self.labels)

        