import torch
import torch.nn as nn
import torch.nn.functional as F

from net import *
import pdb


class focus_net(nn.Module):
    def __init__(self, channel, num_classes, se=True, reduction=2, norm='bn', SOSNet=False, detach=False):
        super(focus_net, self).__init__()
        self.detach = detach
        self.SOSNet = SOSNet
        # downsample twice
        self.conv1x = inconv(channel, 32, norm=norm)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2x = self._make_layer(
            conv_block, 32, 48, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv4x = self._make_layer(
            conv_block, 48, 64, 2, se=se, stride=1, reduction=reduction, norm=norm)
        self.conv4xd2 = self._make_layer(
            conv_block, 64, 64, 2, se=se, stride=1, reduction=reduction, norm=norm, dilation_rate=(2, 2))

        # DenseASPP
        current_num_feature = 64 
        d_feature0 = 64
        d_feature1 = 32
        dropout0 = 0
        self.ASPP_1 = DenseASPPBlock(input_num=current_num_feature, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(2, 2), drop_out=dropout0, norm=norm)

        self.ASPP_2 = DenseASPPBlock(input_num=current_num_feature+d_feature1*1, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(3, 3), drop_out=dropout0, norm=norm)

        self.ASPP_3 = DenseASPPBlock(input_num=current_num_feature+d_feature1*2, num1=d_feature0, num2=d_feature1,
                                     dilation_rate=(4, 4), drop_out=dropout0, norm=norm)

        self.ASPP_4 = DenseASPPBlock(input_num=current_num_feature+d_feature1*3, num1=d_feature0, num2=d_feature1,
                                    dilation_rate=(5, 5), drop_out=dropout0, norm=norm)
        current_num_feature = current_num_feature + 4 * d_feature1

        # upsample
        self.up1 = up_block(in_ch=current_num_feature,
              out_ch=48, se=se, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv2d(48, 48, 3, padding=1)
        # This feature will be used in SOL and SOSNet
        self.up2 = up_block(in_ch=48, out_ch=32, scale=(2, 2), se=se, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv2d(32, 32, 3, padding=1)

        # output branch
        self.out_conv = nn.Conv2d(32, num_classes, 1, 1)

        self.SOL = heatmap_pred(in_ch=32, out_ch=1)

        if self.SOSNet:
            self.SOS = SOSNet_sep(in_ch=32, num=3)


    def forward(self, x, label=False):
        group = 1 
        # down
        o1 = self.conv1x(x)
        o2 = self.maxpool1(o1)
        o2 = self.conv2x(o2)
        o3 = self.maxpool2(o2)
        o3 = self.conv4x(o3)
        o4 = self.conv4xd2(o3)

        # DenseASPP
        aspp1 = self.ASPP_1(o4)
        feature = torch.cat((aspp1, o4), dim=1)

        aspp2 = self.ASPP_2(feature)
        feature = torch.cat((aspp2, feature), dim=1)

        aspp3 = self.ASPP_3(feature)
        feature = torch.cat((aspp3, feature), dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat((aspp4, feature), dim=1)

        #print("feature map: ",feature.size())

        out = self.up1(feature, self.literal1(o2))
        feature_map = self.up2(out, self.literal2(o1))

        out = self.out_conv(feature_map)
        

        if self.detach:
            print('detaching')
            heatmap = self.SOL(feature_map.detach())
        else:
            heatmap = self.SOL(feature_map)

        result = {
            'main_result': out,
            'heatmap': heatmap,
        }
        if self.SOSNet:
            locationList, resultList, labelList = self.SOS(feature_map, heatmap, x, o1, label)
                
            result['small_result'] = resultList
            result['location'] = locationList
            result['small_label'] = labelList

        return result

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


if __name__=='__main__':
    network = focus_net(1, 1, se=True, norm='bn', SOSNet=True)
    ##print(network)
    #exit()
    B = 1
    C = 1
    H = 384
    W = 384
    inputs = torch.randn(B, C, H, W)
    outputs = network(inputs)
    output_1 = outputs['main_result']
    heatmap = outputs['heatmap']
    print(*[outputs[k].size() for k in outputs.keys()])
    print(f'output:{output_1.size()} heatmap:{heatmap.size()}')