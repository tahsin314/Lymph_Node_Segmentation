import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import timm

class Res2Net(nn.Module):
    def __init__(self, model_name = 'res2net50_26w_4s', in_chans = 1):
        super(Res2Net, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.backbone.conv1 = nn.Conv2d(in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)
        return x1, x2, x3, x4

