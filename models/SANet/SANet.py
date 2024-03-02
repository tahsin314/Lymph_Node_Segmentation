#coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from res2net import Res2Net

class SANet(nn.Module):
    def __init__(self, model_name = 'res2net50_26w_4s', in_chans = 1):
        super(SANet, self).__init__()
        # self.args    = args
        self.bkbone  = Res2Net(model_name = model_name, in_chans = in_chans)
        self.linear5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d( 512, 64, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.predict = nn.Conv2d(64*3, 1, kernel_size=1, stride=1, padding=0)
        # self.initialize()

    def forward(self, x, shape=None):
        out2, out3, out4, out5 = self.bkbone(x)
        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)

        out5 = F.interpolate(out5, size=out3.size()[2:], mode='bilinear', align_corners=True)
        out4 = F.interpolate(out4, size=out3.size()[2:], mode='bilinear', align_corners=True)
        pred = torch.cat([out5, out4*out5, out3*out4*out5], dim=1)
        pred = self.predict(pred)
        print([i.size() for i in [out2, out3, out4, out5, pred]])
        return pred

if __name__ == '__main__':
    ras = SANet(in_chans=1).cuda()
    input_tensor = torch.randn(1, 1, 512, 512).cuda()
    out = ras(input_tensor)
    print(out.size())