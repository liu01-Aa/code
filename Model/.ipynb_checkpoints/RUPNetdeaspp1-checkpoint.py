import torch
import torch.nn as nn
from timm.models.layers import  trunc_normal_

import torch.nn.functional as F
import math
class singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(singleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class singleConv0(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(singleConv0, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(single_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(double_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),

            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class doubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,conv_op=nn.Conv2d):
        super(doubleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x

# 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
# 通道注意力
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # kernel_size=1

        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)


class ASPP(nn.Module):

    def __init__(self, inplanes, planes, rate):
        super(ASPP, self).__init__()
        self.rate = rate
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
            # self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, bias=False,padding=1)
            self.conv1 = SeparableConv2d(planes, planes, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu1 = nn.ReLU()

            # self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
            #                         stride=1, padding=padding, dilation=rate, bias=False)
        self.atrous_convolution = SeparableConv2d(inplanes, planes, kernel_size, 1, padding, rate )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.rate != 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
# 空间注意力
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_chs, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attn = channel_attention(in_chs=in_chs, ration=ratio)
        self.spatial_attn = spatial_attention(kernel_size=kernel_size)

    def forward(self, inputs):

        x = self.channel_attn(inputs)
        x = self.spatial_attn(x)

        return x




class cross_conv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_op=nn.Conv2d):
        super(cross_conv, self).__init__()
        # 32, 64, 128, 256
        self.conv0 = singleConv0(in_ch, out_ch, kernel_size=1, stride=2, padding=0, conv_op=nn.Conv2d) # 降维

        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = singleConv(in_ch, in_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d)  # 降维

        self.conv3 = singleConv(in_ch, in_ch, kernel_size=1, stride=2, padding=0, conv_op=conv_op)
        self.conv3_1xk = singleConv(in_ch, in_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), conv_op=conv_op)
        self.conv3_kx1 = singleConv(in_ch, in_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), conv_op=conv_op)

        self.channnel_attn = channel_attention(in_planes=in_ch*3 ,ratio=16)
        # self.spatial_attn = spatial_attention(kernel_size=3)

        self.block = nn.Sequential(nn.Conv2d(in_ch*3, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))


        self.block1 = singleConv0(out_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=conv_op)
        self.conv1x1 = singleConv0(in_ch*3, out_ch, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): #32 256 256
        g = self.conv0(x) # 32*256*256--->64*128*128
        y = self.maxpool(x) # 32*256*256---->32*128*128
        y = self.conv1(y) # 32*128*128

        k = self.conv3(x) # 32*256*256 ---->32*128*128
        k1 = self.conv3_kx1(k) # 32*128*128
        k2 = self.conv3_1xk(k) # 32*128*128

        x = torch.cat([y, k1, k2], dim=1) # 96*128*128
        x = self.channnel_attn(x) * x
        f = x
        x = self.block(x) # 64*128*128
        x = self.relu(x+g)

        f = self.conv1x1(f)
        x = self.block1(x)
        # x = x * self.spatial_attn(x)
        x = self.relu(x+f)

        return x

class Resnet18(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1,conv_op=nn.Conv2d):
        super(Resnet18, self).__init__()
        # [32, 64, 128, 256, 512]
        self.conv = singleConv(in_channels, out_channels[0], kernel_size=kernel_size, stride=stride,
                                 padding=padding,conv_op=conv_op)
        self.cross_0 = cross_conv(out_channels[0], out_channels[1], conv_op=conv_op)
        self.cross_1 = cross_conv(out_channels[1], out_channels[2], conv_op=conv_op)
        self.cross_2 = cross_conv(out_channels[2], out_channels[3], conv_op=conv_op)
        self.cross_3 = cross_conv(out_channels[3], out_channels[4], conv_op=conv_op)


    def forward(self, x):
        feature = []
        x = self.conv(x) #32 * 256 * 256   [32, 64, 160, 352]
        feature.append(x)


        x = self.cross_0(x) # 64 * 128 * 128
        feature.append(x)


        x = self.cross_1(x) # 128 * 64 * 64
        feature.append(x)

        x = self.cross_2(x) # 256 * 32 * 32
        feature.append(x)

        x = self.cross_3(x) # 512 * 16 * 16
        return feature[::-1], x




class R_skip(nn.Module):
    def __init__(self, in_chans=256, out_chans=512,  conv_op=nn.Conv2d):
        super(R_skip, self).__init__()

        self.conv1 = ASPP(in_chans, out_chans, rate=2)
        self.conv2 = ASPP(out_chans, in_chans, rate=5)

        self.conv1x1 = single_conv_relu_batch(out_chans*2, in_chans, 1, 1, 0)

    def forward(self, x): # 256*32*32   512*16*16

        me_x = self.conv1(x)
        me_x_2 = self.conv2(me_x)
        x = torch.cat([x, me_x, me_x_2], dim=1)

        x = self.conv1x1(x)
        return x




class Decoder_UP0(nn.Module):
    def __init__(self, in_chans=1024, out_chans=512, depth=3, conv_op=nn.Conv2d):
        super(Decoder_UP0, self).__init__()
        self.depth = depth
        self.convtrans = nn.ConvTranspose2d(in_channels=in_chans, out_channels=out_chans, kernel_size=2, stride=2)
        self.up0 = double_conv_relu_batch(in_chans*2, out_chans,  kernel_size=3, padding=1, conv_op=conv_op)
    def forward(self, x, f4): # 256*32*32   512*16*16

        y1 = self.convtrans(f4) # 256* 32* 32
        y = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True) #512*32*32
        x = torch.cat([x, y1, y], dim=1) # 512
        f3 = self.up0(x) # 256
        return f3

class Decoder_UP1(nn.Module):
    def __init__(self, in_chans=512, out_chans=256, conv_op=nn.Conv2d):
        super(Decoder_UP1, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels=in_chans, out_channels=out_chans, kernel_size=2, stride=2)
        self.up = double_conv_relu_batch(in_chans*2, out_chans, kernel_size=3, padding=1, conv_op=conv_op)

    def forward(self, x, f3):
        y1 = self.convtrans(f3)  # 256* 32* 32
        y = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat([x, y1, y], dim=1) # 128 x 3
        f2 = self.up(x)
        return f2



class Decoder_UP2(nn.Module):
    def __init__(self, in_chans=256, out_chans=128, conv_op=nn.Conv2d):
        super(Decoder_UP2, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels=in_chans, out_channels=out_chans, kernel_size=2, stride=2)
        self.up = double_conv_relu_batch(in_chans*2, out_chans, kernel_size=3, padding=1,  conv_op=conv_op)

    def forward(self, x, f2):
        y1 = self.convtrans(f2)  # 256* 32* 32
        y = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y1, y], dim=1) # 64 64 32 16
        f1 = self.up(x)
        return f1

class Decoder_UP3(nn.Module):
    def __init__(self, in_chans=128, out_chans=64,  conv_op=nn.Conv2d):
        super(Decoder_UP3, self).__init__()
        self.convtrans = nn.ConvTranspose2d(in_channels=in_chans, out_channels=out_chans, kernel_size=2, stride=2)
        self.up = double_conv_relu_batch(in_chans*2, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
    def forward(self, x, f1):
        y1 = self.convtrans(f1)  # 256* 32* 32
        f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y1, f1], dim=1) #
        f0 = self.up(x)
        return f0


class instanceSegmenationHead(nn.Module):
    def __init__(self, in_chans=54, out_chans=36):
        super(instanceSegmenationHead, self).__init__()

        self.ins_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0)

    def forward(self, y):
        x = self.ins_conv(y)
        return x


class InstanceCounter(nn.Module):
    def __init__(self, input_n_filters, out_chs=16, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters
        self.n_filters = out_chs
        self.out_filter = out_chs
        self.usegpu = usegpu
        self.output = nn.Sequential(nn.Linear(self.out_filter, 1),
                                    nn.Sigmoid())
        self.cnn = nn.Sequential(
                                 single_conv_relu_batch(input_n_filters, self.out_filter, 3, 1, 1),
                                 single_conv_relu_batch(self.out_filter, self.out_filter, 3, 1, 1),
                                nn.AdaptiveAvgPool2d((1, 1))
                                 )


    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x

class RUP_unetdeaspp1(nn.Module):
    def __init__(self,  in_chans=[32, 64, 128, 256, 512], out_chs=16,  out_class=2):
        super(RUP_unetdeaspp1, self).__init__()
        self.down = Resnet18(in_channels=3, out_channels=in_chans)

        self.ins_segHead = instanceSegmenationHead(in_chans[0], out_chans=out_chs)
        self.sem_segHead = instanceSegmenationHead(in_chans[0], out_class)


        # rates = [1, 2, 3, 5]
        # self.aspp1 = ASPP(512, 96, rate=rates[0])
        # self.aspp2 = ASPP(512, 96, rate=rates[1])
        # self.aspp3 = ASPP(512, 96, rate=rates[2])
        # self.aspp4 = ASPP(512, 96, rate=rates[3])
        # self.chann = channel_attention(in_planes=896, ratio=16)

        self.s3 = R_skip(in_chans[3], in_chans[4])
        self.s2 = R_skip(in_chans[2], in_chans[3])
        self.s1 = R_skip(in_chans[1], in_chans[2])
        self.s0 = R_skip(in_chans[0], in_chans[1])

        #
        # self.conv = nn.Sequential(SeparableConv2d(512+384,512,1),
        #                           nn.BatchNorm2d(512),
        #                           nn.ReLU())

        self.up0 = Decoder_UP0(in_chans[4], in_chans[3])
        self.up1 = Decoder_UP1(in_chans[3], in_chans[2])
        self.up2 = Decoder_UP2(in_chans[2], in_chans[1])
        self.up3 = Decoder_UP3(in_chans[1], in_chans[0])

        self.ins_counter = InstanceCounter(in_chans[0], out_chs=out_chs)

        self.apply(self._init_weight)

    def forward(self, x):
        # 编码开始
        features, x = self.down(x)

        # x1 = self.aspp1(x) # 512 --> B X 96 X 16 X 16  (B C H W)
        #
        # x2 = self.aspp2(x) # 512 --> B X 96 X 16 X 16
        #
        # x3 = self.aspp3(x) # 512 --> B X 96 X 16 X 16
        #
        # x4 = self.aspp4(x) # 512 --> B X 96 X 16 X 16
        #
        # # x5 = self.global_avg_pool(x)
        # # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        #
        # x = torch.cat([x, x1, x2, x3, x4], dim=1)
        # x = x * self.chann(x)
        #
        # x = self.conv(x) # 编码结束


        s3 = self.s3(features[0])
        s2 = self.s2(features[1])
        s1 = self.s1(features[2])
        s0 = self.s0(features[3])

        # 解码开始
        f4 = x  # 512 16 16
        f3 = self.up0(s3, f4) # 256 X 32 X 32
        f2 = self.up1(s2, f3)
        f1 = self.up2(s1, f2)
        f0 = self.up3(s0, f1)

        ins_pre = self.ins_segHead(f0)
        seg_pre = self.sem_segHead(f0)
        n_instance = self.ins_counter(f0)
        return  seg_pre, ins_pre, n_instance

    def _init_weight(self, m):
        init_xavier = True
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            if init_xavier:
                    torch.nn.init.xavier_uniform_(m.weight)  # 初始化权重
            else:
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



