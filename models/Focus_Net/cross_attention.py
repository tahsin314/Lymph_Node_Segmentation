import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_layer import Conv
import math


class CrossAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(CrossAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=True)
        )
        self.f_query = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                    kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.key_channels),
                nn.ReLU(inplace=True)
            )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
            kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
            kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x_enc, x_dec):
        batch_size, h, w = x_enc.size(0), x_enc.size(2), x_enc.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x_enc).view(batch_size, self.value_channels, -1)
        # print(f'V before permute:{value.size()}')
        value = value.permute(0, 2, 1)
        query = self.f_query(x_dec).view(batch_size, self.key_channels, -1)
        # print(f'Q before permute:{query.size()}')
        query = query.permute(0, 2, 1)
        key = self.f_key(x_enc).view(batch_size, self.key_channels, -1)
        # print(f'scale:{self.scale} K:{key.size()} Q:{query.size()} V:{value.size()}')
        sim_map = torch.bmm(query, key)
        # print(f"attn_map:{sim_map.size()}")
        sim_map = (self.key_channels**-.5) * sim_map
        
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.bmm(sim_map, value)
        # print(f"context:{context.size()}")
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x_enc.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context