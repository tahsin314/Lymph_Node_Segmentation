<<<<<<< HEAD
# import os
# import numpy as np
# import pandas as pd
# from config.config import config_params

# for key, value in config_params.items():
#     if isinstance(value, str):
#         exec(f"{key} = '{value}'")
#     else:
#         exec(f"{key} = {value}")
        
# df = pd.read_csv(f"{data_dir}/train_labels.csv")
# df = df.drop_duplicates()
# fold = 4
# train_df = df[(df['fold_patient'] != fold)] 
# valid_df = df[df['fold_patient'] == fold]
# test_df = pd.read_csv(f"{data_dir}/test_labels.csv")
# all_df = pd.read_csv(f"{data_dir}/labels.csv")
# all_df = all_df.drop_duplicates()
# # print(len(np.unique(df['patient_id'].tolist())))
# num_train, num_val, num_test, num_all = [], [], [], []

# for i in np.unique(all_df['patient_id'].tolist()):
#     num_all.append(len(os.listdir(f"{data_dir}/{i}/images")))
#     # print(os.listdir(f"{data_dir}/{i}/images/"))
    
# for i in np.unique(train_df['patient_id'].tolist()):
#     num_train.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
# for i in np.unique(valid_df['patient_id'].tolist()):
#     num_val.append(len(os.listdir(f"{data_dir}/{i}/images")))
    
# for i in np.unique(test_df['patient_id'].tolist()):
#     num_test.append(len(os.listdir(f"{data_dir}/{i}/images")))

# # print(num_train, num_val, num_test, num_all, num_train + num_val + num_test)
# print("Comparison of number of slices among train, val and test data")
# print(f"All mean: {np.mean(num_all)} SD: {np.std(num_all)}")
# print(f"Train mean: {np.mean(num_train)} SD: {np.std(num_train)}")
# print(f"Val mean: {np.mean(num_val)} SD: {np.std(num_val)}")
# print(f"Test mean: {np.mean(num_test)} SD: {np.std(num_test)}")

# print(len(all_df))
# print(len(train_df) + len(valid_df) + len(test_df))
# print(len(np.unique(test_df['patient_id'].tolist())))
# print(np.unique(test_df['patient_id'].tolist()))

from multiprocessing import Pool
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config.config import *
from utils import *
from tqdm import tqdm as T
import numpy as np
import cv2
import nrrd
import pandas as pd
# from p_tqdm import p_map
num_slices = 0
data_dir = "../DATA/lymph_node/ct_221"
new_data_dir = f"../DATA/lymph_node/ct_221_{num_slices}_npz"
label_dict = {'patient_id':[], 'slice_num':[], 'label':[]}
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

    # img_pat_id = nrrd.read(os.path.join(data_dir, pat_id, data_file))[0]
    mask_pat_id, metadata = nrrd.read(os.path.join(data_dir, pat_id, seg_file))
    # print(metadata)
    # padded_data = np.pad(img_pat_id, ((0, 0), (0, 0), (num_slices, num_slices)), mode='constant')
    # os.makedirs(os.path.join(new_data_dir, pat_id, 'images'), exist_ok=True)
    # os.makedirs(os.path.join(new_data_dir, pat_id, 'masks'), exist_ok=True)
    ln_lens = []
    len_ln = 0
    for i in range(mask_pat_id.shape[-1]):
        # img = window_image(padded_data[:,:,i-num_slices:i+num_slices+1], 40, 400, 0, 1)
        mask = mask_pat_id[:, :, i]
        mask[mask>0] = 255   
        # label_dict['patient_id'].append(pat_id)
        # label_dict['slice_num'].append(i)
        if np.sum(mask) != 0:
            len_ln += 1
        else:
            if len_ln !=0: ln_lens.append(len_ln)
            len_ln = 0
            
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'images/{i}.png'), img)
        # cv2.imwrite(os.path.join(new_data_dir, pat_id, f'masks/{i}.png'), mask)
    if pat_id == '4793':
        print(ln_lens)
        print(f"Patient ID {pat_id} Mean {np.mean(ln_lens)} SD {np.std(ln_lens)} Max {np.max(ln_lens)} Min {np.min(ln_lens)}")
    return ln_lens

def datapath(patient_id, slice_num):return f"{patient_id}/images/{slice_num}.npz"

if __name__ == '__main__':
    args_list = [(patient_id, num_slices) for patient_id in patient_ids]
    LN_LENS = []
    for arg in T(args_list):
        LN_LENS.extend(data_processing(arg))
        # break
    print(f"Mean {np.mean(LN_LENS)} SD {np.std(LN_LENS)} Max {np.max(LN_LENS)} Min {np.min(LN_LENS)}")
=======
import argparse
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class Bottle2neck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()

        width      = int(math.floor(planes*(baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width*scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs, bns = [], []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = F.relu(self.bns[i](sp), inplace=True)
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


class Res2Net(nn.Module):
    def __init__(self, layers, snapshot, baseWidth=26, scale=4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.snapshot  = snapshot
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(Bottle2neck, 64, layers[0])
        self.layer2 = self._make_layer(Bottle2neck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottle2neck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottle2neck, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load(self.snapshot), strict=False)

def Res2Net50():
    return Res2Net([3, 4, 6, 3], '/home/UFAD/m.tahsinmostafiz/Playground/Lymph_Node_Segmentation/model_dir/res2net50_v1b_26w_4s-3cf99910.pth')

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.PReLU)):
            pass
        else:
            m.initialize()

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args    = args
        self.bkbone  = Res2Net50()
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
        pred = self.predict(pred)
        return pred

    def initialize(self):
        if self.args.snapshot:
            self.load_state_dict(torch.load(self.args.snapshot))
        else:
            weight_init(self)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath'    ,default='../data/train')
    parser.add_argument('--savepath'    ,default='./out')
    parser.add_argument('--mode'        ,default='train')
    parser.add_argument('--lr'          ,default=0.4)
    parser.add_argument('--epoch'       ,default=128)
    parser.add_argument('--batch_size'  ,default=64)
    parser.add_argument('--weight_decay',default=5e-4)
    parser.add_argument('--momentum'    ,default=0.9)
    parser.add_argument('--nesterov'    ,default=True)
    parser.add_argument('--num_workers' ,default=8)
    parser.add_argument('--snapshot'    ,default=None)
    
    args = parser.parse_args()
    ras = Model(args).cuda()
    input_tensor = torch.randn(1, 3, 512, 512).cuda()
    out = ras(input_tensor)
    print(out.size())
>>>>>>> parent of f6c81de... bug fix in data generation
