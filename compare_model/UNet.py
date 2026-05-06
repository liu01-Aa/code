import math

import torch
import torch.nn as nn


def clip_align(x, y):
    deltax = (y.size(2) - x.size(2)) / 2
    deltay = (y.size(3) - x.size(3)) / 2

    if deltax > 0 and deltay > 0:
        y = y[:, :, deltax:-deltax, deltay:-deltay]
    return y


class DownModule(nn.Module):
    """
    Downscale module
    """

    def __init__(self, in_dims, out_dims, repeats=1, padding=0, non_linearity=nn.ELU, use_dropout=False, use_bn=False):
        super(DownModule, self).__init__()
        layers = [nn.Conv2d(in_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]

        for i in range(repeats):
            layers += [nn.Conv2d(out_dims, out_dims, 3, padding=padding)]
            if use_bn:
                layers += [nn.BatchNorm2d(out_dims)]
            layers += [non_linearity(inplace=True)]

        if use_dropout:
            layers += [nn.Dropout2d(p=0.1)]

        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)
        self.non_ln = non_linearity(inplace=True)

    def forward(self, x):
        return self.pool(self.convs(x))


class UpModule(nn.Module):
    """
    Upscale module
    """

    def __init__(self, in_dims, out_dims, repeats=1, padding=0, non_linearity=nn.ELU):
        super(UpModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dims, out_dims, 2, stride=2)
        layers = [nn.Conv2d(2 * out_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]
        for i in range(repeats):
            layers += [nn.Conv2d(out_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]

        self.normconv = nn.Sequential(*[nn.Conv2d(out_dims, out_dims, 2, padding=padding), non_linearity(inplace=True)])
        self.convs = nn.Sequential(*layers)

    def forward(self, x, y):

        x = self.conv(x)

        if 1 == y.size(2) % 2:
            y = self.normconv(y)

        y = clip_align(x, y)

        x = torch.cat([x, y], dim=1)
        return self.convs(x)

class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(single_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class instanceSegmenationHead(nn.Module):
    def __init__(self, in_chans=54, out_chans=36):
        super(instanceSegmenationHead, self).__init__()
        self.ins_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0)

    def forward(self, y):
        x = self.ins_conv(y)
        return x


class InstanceCounter(nn.Module):
    def __init__(self, input_n_filters, out, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters
        self.n_filters = out
        self.out_filter = out
        self.usegpu = usegpu
        self.output = nn.Sequential(nn.Linear(self.out_filter, 1),
                                    nn.Sigmoid())
        self.cnn = nn.Sequential(
                                 nn.Conv2d(self.input_n_filters, self.out_filter, 1, 1, 0),
                                 nn.AdaptiveAvgPool2d((1, 1))
                                 )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x

class feHead(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(feHead, self).__init__()
        self.ins_head = instanceSegmenationHead(in_chans=in_chs, out_chans=out_chs)
        self.seg_head = instanceSegmenationHead(in_chans=in_chs, out_chans=2)
        self.counter = InstanceCounter(input_n_filters=in_chs, out=out_chs)
    def forward(self, x):
        ins_head = self.ins_head(x)
        seg_head = self.seg_head(x)
        count = self.counter(x)
        return seg_head, ins_head, count

class EUnet(nn.Module):
    """
    Deep neural network with skip connections
    """

    def __init__(self, in_dims, out_dims, k=1, s=1, l=1, depth=4, base=32, init_xavier=False, padding=1,
                 non_linearity=nn.ReLU, use_dropout=False, use_bn=True):
        """
        Creates a u-net network
        :param in_dims: input image number of channels
        :param out_dims: number of feature maps
        :param k: width coefficient
        :param s: number of repeats in encoder part
        :param l: number of repeats in decoder part
        """
        super(EUnet, self).__init__()
        self.conv = nn.Conv2d(in_dims, base * k, 3, padding=padding)
        self.head = feHead(in_chs=out_dims, out_chs=10)
        self.depth = depth
        self.down = []
        self.up = []

        for i in range(self.depth):
            dn = DownModule(base * (2 ** i) * k, base * (2 ** (i + 1)) * k, s, non_linearity=non_linearity,
                            padding=padding, use_dropout=use_dropout, use_bn=use_bn)
            up = UpModule(base * (2 ** (i + 1)) * k, base * (2 ** i) * k, l, non_linearity=non_linearity,
                          padding=padding)
            self.add_module("Down" + str(i), dn)
            self.add_module("Up" + str(i), up)
            self.down.append(dn)
            self.up.append(up)

        self.conv1x1 = nn.Conv2d(8 * k, out_dims, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if init_xavier:
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        inter = [self.conv(x)]
        for i in range(self.depth):
            dn = self.down[i](inter[i])
            inter.append(dn)
        # for i in range(len(inter)):
        #     print(inter[i].shape)

        up = inter[-1]
        for i in range(1, self.depth + 1):
            m = self.up[self.depth - i]
            up = m(up, inter[-i - 1])
        # print(up.shape)
        seg_head, ins_head, count=self.head(up)
        return seg_head, ins_head, count
