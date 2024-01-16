import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops
import pdb

from .axial_atten import AA_kernel
# from cross_attention import CrossAttentionBlock
from .self_attention import SelfAttentionBlock
from .conv_layer import Conv


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(net)
    print('Total number of parameters: %d' % num_params)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, kernel_size=1, stride=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1),
                nn.Sigmoid()
                )
    def forward(self, x):
        # print('input: ',x.size())
        y = self.avg_pool(x)
        # print("avg pool size: ",y.size())
        y = self.conv(y)
        # print("after conv: ",y.size())
        # print("x*y: ",(x*y).size())

        return x * y

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1,3,3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, \
                padding=(0,1,1), bias=False, dilation=dilation_rate)

    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
                padding=padding, bias=False, dilation=dilation_rate)

def conv2x2(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,\
            padding=padding, bias=False, dilation=dilation_rate)

class heatmap_pred(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(heatmap_pred, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = conv2x2(in_ch, in_ch//2)

        self.bn2  = nn.BatchNorm2d(in_ch//2)
        self.conv2 = conv2x2(in_ch//2, in_ch//2)

        self.bn3  = nn.BatchNorm2d(in_ch//2)
        self.conv3 = conv2x2(in_ch//2, in_ch//4)

        self.conv4  = nn.Conv2d(in_ch//4, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.conv4(out)

        out = self.sigmoid(out)

        return out

class SOSNet_sep(nn.Module):
    def __init__(self, in_ch, num, pooled_depth=8, pooled_height=64, pooled_width=64, threshold=0.6):
        super(SOSNet_sep, self).__init__()
        layers = []
        for i in range(num):
            layers.append(SOS_branch(1, 1))
        self.SOS_module = nn.ModuleList(layers)
        self.num = num
        self.h_depth = int(pooled_depth//2)
        self.h_height = int(pooled_height//2)
        self.h_width = int(pooled_width//2)
        self.threshold = threshold

    def forward(self, x, heatmap, raw, o1, label=False):
        # print("heatmap size", heatmap.size())
        b, n, h, w = heatmap.shape
        locationList = []

        #all samples will be croped, those sample which doesn't have small organ is negtive samples
        if label is False:
            croped_feature, croped_heatmap, croped_highRes_feat, croped_raw, location, croped_label = self.RoICrop(x, heatmap, raw, o1, label, 0)
        else:
            croped_feature, croped_heatmap, croped_highRes_feat, croped_raw, location, croped_label = self.RoICrop(x, heatmap, raw, o1, label.clone(), 0)


        input_feat = torch.cat((croped_feature, croped_highRes_feat, croped_heatmap, croped_raw), 1)
        result = self.SOS_module[0](input_feat)
        if label is False:
            croped_label = -1
        else:
            croped_label = croped_label
        locationList.append(location)
        for i in range(1, self.num):
            if label is False:
                croped_feature, croped_heatmap, croped_highRes_feat, croped_raw, location, tmp_croped_label = self.RoICrop(x, heatmap, raw, o1, label, i)
            else:
                croped_feature, croped_heatmap, croped_highRes_feat, croped_raw, location, tmp_croped_label = self.RoICrop(x, heatmap, raw, o1, label.clone(), i)

            input_feat = torch.cat((croped_feature, croped_highRes_feat, croped_heatmap, croped_raw), 1)
            tmp_result = self.SOS_module[i](input_feat)
            result = torch.cat((result, tmp_result), dim=1)
            if label is False:
                croped_label = -1
            else:
                croped_label = torch.cat((croped_label, tmp_croped_label), dim=1)
            locationList.append(location)

        return locationList, result, croped_label


    def RoICrop(self, features, heatmap, raw, highRes_feature, label, organ_index):
        label_index = [2, 4, 5]
        b, c, h, w = features.shape

        location = self.center_locate(heatmap[:, :, :])
        roi_z, roi_x, roi_y = location

        croped_feature = features[:, :, roi_z-self.h_depth:roi_z+self.h_depth, roi_x-self.h_height:roi_x+self.h_height, roi_y-self.h_width:roi_y+self.h_width].detach()
        croped_heatmap = heatmap[:, organ_index:organ_index+1, roi_z-self.h_depth:roi_z+self.h_depth, roi_x-self.h_height:roi_x+self.h_height, roi_y-self.h_width:roi_y+self.h_width].detach()
        croped_highRes_feature = highRes_feature[:, :, roi_z-self.h_depth:roi_z+self.h_depth, roi_x-self.h_height:roi_x+self.h_height, roi_y-self.h_width:roi_y+self.h_width].detach()
        croped_raw = raw[:, :, roi_z-self.h_depth:roi_z+self.h_depth, roi_x-self.h_height:roi_x+self.h_height, roi_y-self.h_width:roi_y+self.h_width].detach()

        if not label is False:
            croped_label = label[:, :, roi_z-self.h_depth:roi_z+self.h_depth, roi_x-self.h_height:roi_x+self.h_height, roi_y-self.h_width:roi_y+self.h_width]
            croped_label[croped_label != label_index[organ_index]] = 0
            croped_label[croped_label == label_index[organ_index]] = 1
        else:
            croped_label = -1

        return croped_feature, croped_heatmap, croped_highRes_feature, croped_raw, location, croped_label

    
    
    def gt_center_locate(self, label, label_idx, noise_level=0):
        tmp_label = label[label == label_idx]
        tmp_label = tmp_label[0, 0, :, :, :].cpu().numpy()

        props = regionprops(tmp_label)

        x = self.h_height if x<self.h_height else x
        x = 192-self.h_height if x>192-self.h_height else x
        y = self.h_width if y<self.h_width else y
        y = 192-self.h_width if y>192-self.h_width else y
        z = self.h_depth if z<self.h_depth else z
        z = 40-self.h_depth if z>40-self.h_depth else z
        
        return z, x, y


    def center_locate(self, heatmap):
        b, c, w, h = heatmap.shape
        assert c ==1 ,"Channel should be 1"
        heatmap = heatmap.view(b, -1)
        index = torch.argmax(heatmap, dim=1)
        # print((index))
        y = index//w
        x = index - y*w
        # print(x, y)
        # z = int(index // w // h)
        # index -= z * w * h
        
        # x = int(index // h)
        # index -= x * h

        # y = int(index)

        x = self.h_height if x<self.h_height else x
        x = h-self.h_height if x>h-self.h_height else x
        y = self.h_width if y<self.h_width else y
        y = w-self.h_width if y>w-self.h_width else y
        # z = self.h_depth if z<self.h_depth else z
        # z = 40-self.h_depth if z>40-self.h_depth else z

        return x, y
    





class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv2x2(inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = nn.BatchNorm2d(inplanes)
        elif norm =='in':
            self.bn1 = nn.InstanceNorm2d(inplanes)
        elif norm =='gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv2x2(planes, planes, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = nn.BatchNorm2d(planes)
        elif norm =='in':
            self.bn2 = nn.InstanceNorm2d(planes)
        elif norm =='gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    nn.BatchNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='in':
                self.shortcut = nn.Sequential(
                    nn.InstanceNorm2d(inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, \
                            stride=stride, bias=False)
                )
            elif norm =='gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)
        out += self.shortcut(residue)
        return out

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = SEBasicBlock(out_ch, out_ch, kernel_size=(3,3), norm=norm)
    def forward(self, x): 
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out 

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out
class ReverseAxialAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ReverseAxialAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        # self.out_conv = nn.Conv2d(in_ch, 1, 1, 1)
        self.out_conv = nn.Sequential(
                            nn.Conv2d(in_ch, in_ch//2, 3, 1, 1),
                            nn.Conv2d(in_ch//2, 1, 1, 1))
        self.aa_kernel = AA_kernel(out_ch, out_ch)

        self.ra_conv1 = Conv(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv2 = Conv(out_ch,out_ch,3,1,padding=1,bn_acti=True)
        self.ra_conv3 = Conv(out_ch,1,3,1,padding=1,bn_acti=True)

    def forward(self, dec_out, enc_out):
        partial_output = self.out_conv(dec_out)
        partial_output_ra = -1*(torch.sigmoid(partial_output)) + 1

        aa_attn = self.aa_kernel(enc_out)
        aa_attn_o = partial_output_ra.expand(-1, self.out_ch, -1, -1).mul(aa_attn)

        ra =  self.ra_conv1(aa_attn_o) 
        ra = self.ra_conv2(ra) 
        ra = self.ra_conv3(ra)

        out = ra + partial_output

        return out


class up_block_cross_attn(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2, 2), se=False, reduction=2, norm='bn'):
        super(up_block_cross_attn, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )
        # self.ra_attn = ReverseAxialAttention(in_ch+out_ch, out_ch)

        # self.ca = CrossAttentionBlock(in_channels=out_ch, key_channels=out_ch//2, value_channels=out_ch//4 )
        self.attn_block = SelfAttentionBlock(in_channels=in_ch+out_ch, key_channels=out_ch, value_channels=out_ch//2 )

    def sparse_attention(self, x):
        N, C, H, W = x.shape
        p_h = p_w = 12
        q_h = H // p_h
        q_w = W // p_w
        x_r = x.reshape(N, C, q_h, p_h, q_w, p_w)
        x_p = x_r.permute(0, 3, 5, 1, 2, 4)
        x = x_p.reshape(N * p_h * p_w, C, q_h, q_w)
       
        global_relation = self.attn_block(x)
        
        gr_r = global_relation.reshape(N, p_h, p_w, C, q_h, q_w)
        gr_p = gr_r.permute(0, 4, 5, 3, 1, 2)
        gr = gr_p.reshape(N * q_h * q_w, C, p_h, p_w)
        attn_out = self.attn_block(gr)

        x = attn_out.reshape(N, q_h, q_w, C, p_h, p_w)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        return x

    def forward(self, x_dec, x_enc):  #x1 from dec and x2 fro encoder
        x_dec = F.interpolate(x_dec, scale_factor=self.scale, mode='nearest')
        out = torch.cat([x_enc, x_dec], dim=1)
        out = self.sparse_attention(out)
        out = self.conv(out)
        return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2, 2), se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        self.scale = scale

        self.conv = nn.Sequential(
            conv_block(in_ch+out_ch, out_ch, se=se, reduction=reduction, norm=norm)
        )
        self.ra_attn = ReverseAxialAttention(in_ch+out_ch, out_ch)

        # self.ra_attn = ReverseAxialAttention(in_ch, out_ch)

    def forward(self, x_dec, x_enc):  #x1 from dec and x2 fro encoder
        x_dec = F.interpolate(x_dec, scale_factor=self.scale, mode='nearest')
        # print(f'x_enc:{x_enc.shape} x_dec:{x_dec.shape}')
        out = torch.cat([x_enc, x_dec], dim=1)
        ra_out = self.ra_attn(out, x_enc)    #with concatenated feature
        # ra_out = self.ra_attn(x_dec, x_enc)      #with only decoder feature
        out = self.conv(out)
        return out, ra_out

class up_nocat(nn.Module):
    def __init__(self, in_ch, out_ch, scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_nocat, self).__init__()

        self.scale = scale
        self.conv = nn.Sequential(
            conv_block(out_ch, out_ch, se=se, reduction=reduction, norm=norm),
        )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=True)
        out = self.conv(out)

        return out

class literal_conv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, reduction=2, norm='bn'):
        super(literal_conv, self).__init__()
        self.conv = conv_block(in_ch, out_ch, se=se, reduction=reduction, norm=norm)
    def forward(self, x):
        out = self.conv(x)
        return out

class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', nn.BatchNorm2d(input_num))
            elif norm == 'in':
                self.add_module('norm_1', nn.InstanceNorm2d(input_num))
            elif norm == 'gn':
                self.add_module('norm_1', nn.GroupNorm(NUM_GROUP, input_num))

        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', nn.BatchNorm2d(num1))
        elif norm == 'in':
            self.add_module('norm_2', nn.InstanceNorm2d(num1))
        elif norm == 'gn':
            self.add_module('norm_2', nn.GroupNorm(NUM_GROUP, num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(feature, p=self.drop_rate, training=self.training)

        return feature


class SOS_branch(nn.Module):
    """share weights before the last conv layer"""
    def __init__(self, channel, num_classes, se=True, reduction=2, norm='bn'):
        super(SOS_branch, self).__init__()
        # downsample twice
        self.share_conv1x = inconv(channel, 24, norm=norm)

        self.share_conv1x_2 = self._make_layer(
            conv_block,  24, 32, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.share_maxpool1 = nn.MaxPool2d((2, 2))

        self.share_conv2x = self._make_layer(
            conv_block, 32, 48, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.share_maxpool2 = nn.MaxPool2d((2, 2)) 

        self.share_conv4x = self._make_layer(
            conv_block, 48, 64, 2, se=se, stride=1, reduction=reduction, norm=norm)

        # DenseASPP
        current_num_feature = 64
        d_feature0 = 64
        d_feature1 = 32
        dropout0 = 0 
        self.share_ASPP_1 = DenseASPPBlock(input_num=current_num_feature, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 3, 3), drop_out=dropout0, norm=norm)

        self.share_ASPP_2 = DenseASPPBlock(input_num=current_num_feature+d_feature1*1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 5, 5), drop_out=dropout0, norm=norm)

        self.share_ASPP_3 = DenseASPPBlock(input_num=current_num_feature+d_feature1*2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 7, 7), drop_out=dropout0, norm=norm)

        self.share_ASPP_4 = DenseASPPBlock(input_num=current_num_feature+d_feature1*3, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(1, 9, 9), drop_out=dropout0, norm=norm)
        current_num_feature = current_num_feature + 4 * d_feature1

        # upsample
        self.share_up1 = up_block(in_ch=current_num_feature,
                               out_ch=48, se=se, reduction=reduction, norm=norm)
        self.share_literal1 = nn.Conv2d(48, 48, 3, padding=1)

        self.share_up2 = up_block(in_ch=48, out_ch=32, scale=(
            1, 2, 2), se=se, reduction=reduction, norm=norm)
        self.share_literal2 = nn.Conv2d(32, 32, 3, padding=1)
        # branch
        self.out_conv = nn.Conv2d(32, num_classes, 1, 1)



    def forward(self, x):
        # down
        x1 = self.share_conv1x(x[:, -1:, :, :, :])

        o1 = self.share_conv1x_2(x1)

        o2 = self.share_maxpool1(o1)
        o2 = self.share_conv2x(o2)
        o3 = self.share_maxpool2(o2)
        o3 = self.share_conv4x(o3)

        # DenseASPP
        aspp1 = self.share_ASPP_1(o3)
        feature = torch.cat((aspp1, o3), dim=1)

        aspp2 = self.share_ASPP_2(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.share_ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.share_ASPP_4(feature)
        feature = torch.cat((aspp4, feature), dim=1)

        out = self.share_up1(feature, self.share_literal1(o2))
        out = self.share_up2(out, self.share_literal2(o1))

        out = self.out_conv(out)

        return out


    def _make_layer(self, block, in_ch, out_ch, num_blocks, se=True, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, se=se, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, se=se, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))

        return nn.Sequential(*layers)

        out = self.out_conv(out)

        return out
