import torch
import torch.nn as nn
from torch.nn import functional as F
from torchviz import make_dot
from pytorch_model_summary import summary


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = torch.cat([x, out], 1)
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.pool(out)
        return out

class DenseNet1D(nn.Module):
    def __init__(self, in_channels, growth_rate, block_config, num_classes):
        super(DenseNet1D, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('conv0', nn.Conv1d(in_channels, 64, kernel_size=7, padding=3))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool1d(kernel_size=2, stride=2))

        num_features = 64
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionBlock(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool1d(features, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def densenet1d(num_channels, num_classes):
    return DenseNet1D(in_channels=num_channels, growth_rate=6, block_config=[4, 4, 4], num_classes=num_classes)

if __name__ == "__main__":
    # Example usage:
    num_classes = 10
    model = densenet1d(3, num_classes)  # Corrected in_channels to 3
    data = torch.randn(4, 3, 9000)
    out = model(data)
    print(out.shape)

    print(summary(model, data, show_input=False))
    dot = make_dot(model(data), params=dict(model.named_parameters()))
    dot.format = 'png'  # You can change the format as needed
    dot.render('../model_graphs/densenet1D_graph')
