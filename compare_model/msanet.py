from tkinter import E
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

# import Constants

nonlinearity = partial(F.relu, inplace=True)


class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
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
    def __init__(self, in_chs=54, out_chs=36):
        super(instanceSegmenationHead, self).__init__()
        self.ins_conv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, 1, 1, 0))

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
            single_conv_relu_batch(self.input_n_filters, self.n_filters, kernel_size=3, padding=1),
            nn.Conv2d(self.n_filters, self.out_filter, 1, 1, 0),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x


class feHead(nn.Module):
    def __init__(self, in_chs=19, out_chs=12):
        super(feHead, self).__init__()

        self.ins_counter = InstanceCounter(in_chs, out=out_chs)  # 计算实例数量
        self.ins_segHead = instanceSegmenationHead(in_chs, out_chs=out_chs)
        self.sem_segHead = instanceSegmenationHead(in_chs, 2)

    def forward(self, x):
        # print(x.shape)

        ins_head = self.ins_segHead(x)
        seg_head = self.sem_segHead(x)
        count = self.ins_counter(x)
        return seg_head, ins_head, count
class MACblock(nn.Module):
    def __init__(self, channel):
        super(MACblock, self).__init__()
        self.dilate1 = nn.ConvTranspose2d(channel, channel, kernel_size=3, dilation=1, padding=1,output_padding=0)
        self.dilate2 = nn.ConvTranspose2d(channel, channel, kernel_size=3, dilation=4, padding=4,output_padding=0)
        self.dilate3 = nn.ConvTranspose2d(channel, channel, kernel_size=3, dilation=6, padding=6,output_padding=0)
        self.conT1x1 = nn.ConvTranspose2d(channel, channel, kernel_size=1, dilation=1, padding=0,output_padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conT1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conT1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conT1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=1,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=1,padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=1,padding=2)

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear',align_corners=True)
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear',align_corners=True)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class MSANET(nn.Module):
    def __init__(self, num_classes=3, num_channels=3,out_classes=12):
        super(MSANET, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.droput=nn.Dropout()
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = MACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.final_super = feHead(in_chs=num_channels, out_chs=out_classes)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.droput(e4)
        e4 = self.spp(e4)
        e4 = self.droput(e4)
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        seg_head, ins_head, count = self.final_super(out)
        return seg_head, ins_head, count
#
# if __name__=="__main__":
#
#     x = torch.rand(16,3,320,160)
#     ds = MSANET()
#     x1,x2,x3 = ds(x)
#     print(x1.shape)
#     print(x2.shape)
#     print(x3.shape)