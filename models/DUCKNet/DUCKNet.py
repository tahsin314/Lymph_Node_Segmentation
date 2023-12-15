import torch
import torch.nn as nn
import torch.nn.functional as F

from .ConvBlock2D import ConvBlock2D

class DuckNet(nn.Module):
    def __init__(self, in_chans, starting_filters=17):
        super(DuckNet, self).__init__()

        self.p1 = nn.Conv2d(in_chans, starting_filters * 2, kernel_size=2, stride=2, padding='valid')
        self.p2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding='valid')
        self.p3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding='valid')
        self.p4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding='valid')
        self.p5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding='valid')

        self.t0 = ConvBlock2D(in_chans, starting_filters, 'duckv2', repeat=1)

        self.l1i = nn.Conv2d(starting_filters, starting_filters * 2, kernel_size=2, stride=2, padding='valid')
        self.t1 = ConvBlock2D(starting_filters*2, starting_filters * 2, 'duckv2', repeat=1)

        self.l2i = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding='valid')
        self.t2 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)

        self.l3i = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding='valid')
        self.t3 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)

        self.l4i = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding='valid')
        self.t4 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)

        self.l5i = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding='valid')
        self.t51 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.t53 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=1)
        self.t54 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'resnet', repeat=1)

        self.l5o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q4 = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)

        self.l4o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q3 = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)

        self.l3o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q6 = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)

        self.l2o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q1 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)

        self.l1o = nn.Upsample(scale_factor=2, mode='nearest')
        self.q0 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        self.z1 = ConvBlock2D(starting_filters, starting_filters, 'duckv2', repeat=1)

        self.output = nn.Conv2d(starting_filters, 1, kernel_size=1, stride=1)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        t0 = self.t0(x)

        l1i = self.l1i(t0)
        s1 = l1i + p1
        t1 = self.t1(s1)

        l2i = self.l2i(t1)
        s2 = l2i + p2
        t2 = self.t2(s2)

        l3i = self.l3i(t2)
        s3 = l3i + p3
        t3 = self.t3(s3)

        l4i = self.l4i(t3)
        s4 = l4i + p4
        t4 = self.t4(s4)
        l5i = self.l5i(t4)
        s5 = l5i + p5
        t51 = self.t51(s5)
        t53 = self.t53(t51)
        t54 = self.t54(t53)
        l5o = self.l5o(t54)
        c4 = l5o + t4
        q4 = self.q4(c4)

        l4o = self.l4o(q4)
        c3 = l4o + t3
        q3 = self.q3(c3)

        l3o = self.l3o(q3)
        c2 = l3o + t2
        q6 = self.q6(c2)

        l2o = self.l2o(q6)
        c1 = l2o + t1
        q1 = self.q1(c1)

        l1o = self.l1o(q1)
        c0 = l1o + t0
        z1 = self.q0(c0)

        output = self.output(z1)
        return output

if __name__ == '__main__':
    # ras = ResNetConv2DBlock
    ras = DuckNetV2(in_chans=1, starting_filters=34).cuda()
    # print(ras)
    input_tensor = torch.randn(1, 1, 512, 512).cuda()
    out = ras(input_tensor)
    print(out.size())