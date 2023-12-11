import torch
import torch.nn as nn
from torchviz import make_dot
from pytorch_model_summary import summary


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet1D(nn.Module):
    def __init__(self, input_channels=1, num_classes=1000, width_multiplier=1.0):
        super(MobileNet1D, self).__init__()
        channels = [32, 64, 128, 128, 256, 256, 512]
        channels = [int(c * width_multiplier) for c in channels]

        self.conv1 = nn.Conv1d(input_channels, channels[0], kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()

        for i in range(1, len(channels)):
            self.layers.append(self._make_layer(channels[i - 1], channels[i], 2 if i == 1 else 1))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = [DepthwiseSeparableConv1D(in_channels, out_channels, stride=stride),
                  nn.ReLU(inplace=True),
                  DepthwiseSeparableConv1D(out_channels, out_channels),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Example usage:
    num_channels = 3
    num_classes = 3
    model = MobileNet1D(num_channels, num_classes)

    # model = resnet1d(3, num_classes)
    data = torch.randn(4, 3, 900)
    out = model(data)
    print(out.shape)
    print(summary(model, data, show_input=False))
    dot = make_dot(model(data), params=dict(model.named_parameters()))
    dot.format = 'png'  # You can change the format as needed
    dot.render('../model_graphs/mobileresnet1D_graph')