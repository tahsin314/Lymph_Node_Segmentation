# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:17:13 2021

@author: angelou
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_layer import Conv3D
from self_attention import self_attn_3d
import math

class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv3D(in_channel, out_channel, kSize=(1,1,1),stride=(1,1,1),padding=0)
        self.conv1 = Conv3D(out_channel, out_channel, kSize=(3, 3, 3),stride = (1,1,1), padding=(1,1,1))
        self.Dattn = self_attn_3d(out_channel, mode='d')
        self.Hattn = self_attn_3d(out_channel, mode='h')
        self.Wattn = self_attn_3d(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        #print(f'inside AA kernel::: x after conv0 and 1: {x.shape}')
        # exit()
        Dx = self.Dattn(x)
        Hx = self.Hattn(Dx)
        Wx = self.Wattn(Hx)

        return Wx