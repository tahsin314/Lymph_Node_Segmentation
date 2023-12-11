import torch
import torch.nn as nn
try:
    from .utils import Head
except:
    from utils import Head

from torchviz import make_dot
from pytorch_model_summary import summary

import timm
from pprint import pprint

model_names = timm.list_models('*seresnet*')
print(model_names)
# print(torch.hub.list('zhanghang1989/ResNeSt', force_reload=True))
class Resne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=1):
        super().__init__()
        # self.backbone = timm.create_model(model_name, pretrained=True)
        # self.backbone = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=True)
        # print(self.backbone)
        self.backbone = timm.create_model(model_name, pretrained=True)
        # print(self.backbone)
        # self.in_features = 2048
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.out = nn.Linear(self.in_features, num_class)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.head(x)
        return x
if __name__ == "__main__":
    # Example usage:
    num_classes = 10
    model = Resne_t('resnet34', num_classes)
    data = torch.randn(4, 3, 256, 256)
    out = model(data)
    print(out.shape)
    print(summary(model, data, show_input=False))
    dot = make_dot(model(data), params=dict(model.named_parameters()))
    dot.format = 'png'  # You can change the format as needed
    dot.render('../model_graphs/resnet34_graph')
