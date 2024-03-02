import torch
import torch.nn as nn

kernel_initializer = 'he_uniform'


class Conv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='relu', padding='same'):
        super(Conv2DLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation == 'relu' else None

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock2D(nn.Module):
    def __init__(self, in_chans, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
        super(ConvBlock2D, self).__init__()
        self.in_chans = in_chans
        self.filters = filters
        self.block_type = block_type
        self.repeat = repeat
        self.dilation_rate = dilation_rate
        self.size = size
        self.padding = padding
        self.SeparatedConv2DBlock = SeparatedConv2DBlock(self.in_chans, self.filters, self.size, self.padding)
        self.DuckV2Conv2DBlock = DuckV2Conv2DBlock(self.in_chans, self.filters, self.size)
        self.MidscopeConv2DBlock = MidscopeConv2DBlock(self.in_chans, self.filters)
        self.WidescopeConv2DBlock = WidescopeConv2DBlock(self.in_chans, self.filters)
        self.ResNetConv2DBlock = ResNetConv2DBlock(self.in_chans, self.filters, self.dilation_rate)
        self.Conv2DLayer = Conv2DLayer(self.in_chans, self.filters, self.size, activation='relu', padding=self.padding)
        self.DoubleConvolutionWithBatchNormalization = DoubleConvolutionWithBatchNormalization(self.in_chans, self.filters, self.dilation_rate)
    
    def forward(self, x):
        for i in range(self.repeat):
            if self.block_type == 'separated':
                x = self.SeparatedConv2DBlock(x)
            elif self.block_type == 'duckv2':
                x = self.DuckV2Conv2DBlock(x)
            elif self.block_type == 'midscope':
                x = self.MidscopeConv2DBlock(x)
            elif self.block_type == 'widescope':
                x = self.WidescopeConv2DBlock(x)
            elif self.block_type == 'resnet':
                x = self.ResNetConv2DBlock(x)
            elif self.block_type == 'conv':
                x = self.Conv2DLayer(x)
            elif self.block_type == 'double_convolution':
                x = self.DoubleConvolutionWithBatchNormalization(x)
            else:
                return None

        return x


class SeparatedConv2DBlock(nn.Module):
    def __init__(self, in_chans, filters, size=3, padding='same', kernel_initializer='he_uniform'):
        super(SeparatedConv2DBlock, self).__init__()
        self.in_chans = in_chans
        self.conv1 = nn.Conv2d(in_chans, filters, kernel_size=(1, size), padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(size, 1), padding=padding)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        return x


class MidscopeConv2DBlock(nn.Module):
    def __init__(self, in_chans, filters, kernel_initializer='he_uniform'):
        super(MidscopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, filters, kernel_size=(3, 3), padding='same', dilation=1)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=2)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        return x


class WidescopeConv2DBlock(nn.Module):
    def __init__(self, in_chans, filters, kernel_initializer='he_uniform'):
        super(WidescopeConv2DBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, filters, kernel_size=(3, 3), padding='same', dilation=1)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=2)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=3)
        self.batch_norm3 = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activation(x)
        return x


class ResNetConv2DBlock(nn.Module):
    def __init__(self, in_chans, filters, dilation_rate=1, kernel_initializer='he_uniform'):
        super(ResNetConv2DBlock, self).__init__()
        self.in_chans = in_chans
        self.conv1 = nn.Conv2d(in_chans, filters, kernel_size=(1, 1), padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv2d(in_chans, filters, kernel_size=(3, 3), padding='same', dilation=dilation_rate)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=dilation_rate)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        # print(x.size(), self.in_chans)
        x1 = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x_final = x + x1
        x_final = self.activation(x_final)
        return x_final


class DoubleConvolutionWithBatchNormalization(nn.Module):
    def __init__(self, in_chans, filters, dilation_rate=1, kernel_initializer='he_uniform'):
        super(DoubleConvolutionWithBatchNormalization, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, filters, kernel_size=(3, 3), padding='same', dilation=dilation_rate)
        self.batch_norm1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), padding='same', dilation=dilation_rate)
        self.batch_norm2 = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        return x

class DuckV2Conv2DBlock(nn.Module):
    def __init__(self, in_chans, filters, size):
        super(DuckV2Conv2DBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=filters)
        self.widescope_block = WidescopeConv2DBlock(in_chans, filters)
        self.midscope_block = MidscopeConv2DBlock(in_chans, filters)
        self.resnet_block1 = ResNetConv2DBlock(in_chans, filters, dilation_rate=1)
        self.resnet_block2 = ResNetConv2DBlock(in_chans, filters, dilation_rate=2)
        self.resnet_block3 = ResNetConv2DBlock(in_chans, filters, dilation_rate=3)
        self.separated_block = SeparatedConv2DBlock(in_chans, filters, size=size)

    def forward(self, x):
        x1 = self.widescope_block(x)
        x2 = self.midscope_block(x)
        x3 = self.resnet_block1(x)
        x4 = self.resnet_block2(x)
        x5 = self.resnet_block3(x)
        x6 = self.separated_block(x)

        x = x1 + x2 + x3 + x4 + x5 + x6

        x = self.batch_norm(x)
        return x
