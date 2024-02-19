# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:15:44 2021

@author: angelou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_layer import Conv, Conv3D
import math


class SelfAttentionBlock(nn.Module):
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
        super(SelfAttentionBlock, self).__init__()
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


    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        # #print(f'V before permute:{value.size()}')
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        # #print(f'Q before permute:{query.size()}')
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        # #print(f'scale:{self.scale} K:{key.size()} Q:{query.size()} V:{value.size()}')
        sim_map = torch.bmm(query, key)
        # #print(f"attn_map:{sim_map.size()}")
        sim_map = (self.key_channels**-.5) * sim_map
        
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.bmm(sim_map, value)
        # #print(f"context:{context.size()}")
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context
    
class self_attn_3d(nn.Module):
    def __init__(self, in_channels, mode='dhw'):
        super(self_attn_3d, self).__init__()

        self.mode = mode

        # Adjusting for 3D convolution
        self.query_conv = Conv3D(in_channels, in_channels // 8, kSize=(1, 1, 1), stride=1, padding=0)
        self.key_conv = Conv3D(in_channels, in_channels // 8, kSize=(1, 1, 1), stride=1, padding=0)
        self.value_conv = Conv3D(in_channels, in_channels, kSize=(1, 1, 1), stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, depth, height, width = x.size()
        #print(f'################# mode attention: {self.mode} ############################')
        axis = 1
        if 'd' in self.mode:
            axis *= depth
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        # Adjusting query, key, value projections for 3D
        query = self.query_conv(x)
        projected_query = query.view(*view).permute(0, 2, 1)
        #print(f'q:{query.size()} proj_query:{projected_query.size()}')

        key = self.key_conv(x)
        projected_key = key.view(*view)
        #print(f'k:{key.size()} proj_key:{projected_key.size()}')

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        #print(f'attn:{attention.size()}')
        projected_value = self.value_conv(x).view(*view)
        #print(f'proj_val:{projected_value.size()}')

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        #print(f'attn_out:{out.size()}')
        # Reshaping the output back to the original 3D input shape
        out = out.view(batch_size, channel, depth, height, width)
        #print(f'attn_out_reshape:{out.size()}')

        out = self.gamma * out + x
        return out

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()
        #print(f'################# mode attention: {self.mode} ############################')
        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)
        # #print(f'view:{view}')
        query = self.query_conv(x)
        
        projected_query = query.view(*view).permute(0, 2, 1)
        #print(f'q:{query.size()} proj_query:{projected_query.size()}')

        key = self.key_conv(x)
        projected_key = key.view(*view)
        #print(f'k:{key.size()} proj_key:{projected_key.size()}')

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        #print(f'attn:{attention.size()}')
        projected_value = self.value_conv(x).view(*view)
        #print(f'proj_val:{projected_value.size()}')

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        #print(f'attn_out:{out.size()}')
        out = out.view(batch_size, channel, height, width)
        #print(f'attn_out_reshape:{out.size()}')

        out = self.gamma * out + x
        return out

if __name__=="__main__":
    attn = SelfAttentionBlock(in_channels=48, key_channels=24, value_channels=12)
    # x = torch.randn(8, 48, 96, 96)
    # #print("height Attention")
    # AttnLayer = self_attn(48, mode='h')
    # out = AttnLayer(x)
    # #print("width Attention")
    # AttnLayer = self_attn(48, mode='w')
    # out = AttnLayer(x)
    x_dec = torch.randn(2, 48, 8, 8)
    N, C, H, W = x_dec.shape
    p_h = p_w = 4
    q_h = H//p_h
    q_w = W//p_w

    x_r = x_dec.reshape(N, C, q_h, p_h, q_w, p_w)
    x_p = x_r.permute(0, 3, 5, 1, 2, 4)
    x = x_p.reshape(N * p_h * p_w, C, q_h, q_w)

    #print(f"x_reshape:{x_r.shape} x_p:{x_p.size()} x:{x.size()}")

    
    global_relation = attn(x)
    #print('global_relation: ',global_relation.size())

    gr_r = global_relation.reshape(N, p_h, p_w, C, q_h, q_w)
    gr_p = gr_r.permute(0, 4, 5, 3, 1, 2)
    gr = gr_p.reshape(N * q_h * q_w, C, p_h, p_w)
    x = attn(gr)

    #print(f"gr_r:{gr_r.shape} gr_p:{gr_p.size()} gr:{gr.size()}")

    #print('local relation: ',x.size())
