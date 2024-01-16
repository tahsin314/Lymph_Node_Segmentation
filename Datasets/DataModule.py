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

class DataModule():
    def __init__(self, train_ds, valid_ds, test_ds, 
    batch_size=32, sampler=None, shuffle=True, num_workers=8):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        if "Windows" in system():
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        self.sampler = sampler

    def train_dataloader(self):
        if self.sampler is not None:
            sampler = self.sampler(labels=self.train_ds.get_labels(), mode="upsampling")
            train_loader = DataLoader(self.train_ds,batch_size=self.batch_size, 
            sampler= sampler, shuffle=False, drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle, 
            drop_last=True,
            num_workers=self.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_ds,batch_size=self.batch_size, drop_last=True,
        shuffle=True,
         num_workers=self.num_workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        if self.test_ds is not None:
            test_loader = DataLoader(self.test_ds,batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True)
        return test_loader