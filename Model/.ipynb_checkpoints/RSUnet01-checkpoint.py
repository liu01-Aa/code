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
            single_conv_relu_batch(in_ch, out_ch, kernel_size=1, stride=1, padding=0, conv_op=conv_op),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
         #32, 64, 128, 256
        self.conv0 = singleConv0(in_ch, out_ch, kernel_size=1, stride=2, padding=0, conv_op=nn.Conv2d) # 降维

        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(singleConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d),
                                   singleConv(out_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d)) # 降维


        self.conv3 = singleConv(in_ch, out_ch, kernel_size=2, stride=2, padding=0, conv_op=conv_op)
        self.conv3_1= singleConv(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), conv_op=conv_op)
        self.conv3_2 = singleConv(out_ch, out_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), conv_op=conv_op)

        self.channnel_attn = channel_attention(in_planes=out_ch*3, ratio=16)


        self.block = nn.Sequential(nn.Conv2d(out_ch*3, in_ch, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(in_ch),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch))


        self.block1 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_ch)
                                    )
        self.conv1x1 = singleConv0(out_ch*3, out_ch, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        g = self.conv0(x)

        y = self.maxpool(x)
        y = self.conv1(y)

        k = self.conv3(x)
        k1 = self.conv3_1(k)
        k2 = self.conv3_2(k)

        x = torch.cat([y, k1, k2], dim=1)
        x = self.channnel_attn(x) * x
        f = x
        x = self.block(x)
        x = self.relu(x+g)

        f = self.conv1x1(f)
        x = self.block1(x)

        x = self.relu(x+f)

        return x

class Resnet18(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=(3, 3), stride=1, padding=1,conv_op=nn.Conv2d):
        super(Resnet18, self).__init__()
         #[64, 128, 256, 512]
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


        x = self.cross_0(x)
        feature.append(x)


        x = self.cross_1(x)
        feature.append(x)

        x = self.cross_2(x)
        feature.append(x)

        x = self.cross_3(x)
        return feature[::-1], x    #将存储有每个阶段特征的列表 feature逆序返回，并同时返回最后一阶段的输出x

class instanceSegmenationHead(nn.Module):
    def __init__(self, in_chans=54, out_chans=36):
        super(instanceSegmenationHead, self).__init__()
        self.ins_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0)
    def forward(self, y):
        x = self.ins_conv(y)
        return x


class RC_skip(nn.Module):
    def __init__(self, in_chans=256, out_chans=128,  conv_op=nn.Conv2d):
        super(RC_skip, self).__init__()
        self.convtrans_d3 = nn.ConvTranspose2d(in_chans*2, out_chans*2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d2_1 = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, padding=0)
        self.convtrans_d2_2 = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, padding=0)

        self.convtrans_d1_1 = nn.ConvTranspose2d(out_chans, out_chans//2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d1_2 = nn.ConvTranspose2d(out_chans, out_chans//2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d1_3 = nn.ConvTranspose2d(out_chans, out_chans//2, kernel_size=2, stride=2, padding=0)

        self.up_d2_1 =  single_conv_relu_batch(in_chans, out_chans,  kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d2_2=  single_conv_relu_batch(in_chans, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d3 =  single_conv_relu_batch(in_chans*2, in_chans, kernel_size=3, padding=1, conv_op=conv_op)
        self.conv1x1 = single_conv_relu_batch(in_chans, out_chans//2, kernel_size=1, stride=1, padding=0, conv_op=conv_op)

    def forward(self, d1, d2, d3, d4):  # d2, d3, d4 网络深度节点输出
        d4 = self.convtrans_d3(d4)
        d3_1 = torch.cat([d3, d4], dim=1)  # 512
        d3_1 = self.up_d3(d3_1)

        d3 = self.convtrans_d2_1(d3)
        d2_1 = torch.cat([d2, d3], dim=1)  # 512
        d2_1 = self.up_d2_1(d2_1)

        d2_2= self.convtrans_d2_2(d3_1)
        d2_2 = torch.cat([d2, d2_2], dim=1)  # 512
        d2_2 = self.up_d2_2(d2_2)

        d1_1 = self.convtrans_d1_1(d2)
        d1_2 = self.convtrans_d1_2(d2_1)
        d1_3 = self.convtrans_d1_3(d2_2)


        x = torch.cat([d1, d1_1, d1_2, d1_3], dim=1)
        x = self.conv1x1(x)

        return x



class Decoder_UP0(nn.Module):
    def __init__(self, in_chans=1024, out_chans=512,  conv_op=nn.Conv2d):
        super(Decoder_UP0, self).__init__()

        self.up0 = double_conv_relu_batch(in_chans+out_chans, out_chans,  kernel_size=3, padding=1, conv_op=conv_op)
    def forward(self, x, f4):
        y = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1) # 512
        f3 = self.up0(x) # 256
        return f3

class Decoder_UP1(nn.Module):
    def __init__(self, in_chans=512, out_chans=256, conv_op=nn.Conv2d):
        super(Decoder_UP1, self).__init__()

        self.up = double_conv_relu_batch(in_chans+out_chans, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
        self.conv1x1 = single_conv_relu_batch(in_chans*2, out_chans, kernel_size=1, stride=1, padding=0, conv_op=conv_op)

    def forward(self, f3, f4):
        f3 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)   #对f3进行上采样，将其尺寸增加一倍，使用双线性插值法，并确保角点对齐

        f4 = self.conv1x1(f4)

        f4 = F.interpolate(f4, scale_factor=4, mode='bilinear', align_corners=True)  #对 f4进行上采样，将其尺寸增加四倍，使用双线性插值法，并确保角点对齐

        x = torch.cat([f3, f4], dim=1) # 128 x 3
        f2 = self.up(x)      #将拼接后的特征 x传递给双卷积层进行特征提取和变换，得到最终的特征
        return f2


class Decoder_UP2(nn.Module):
    def __init__(self, in_chans=256, out_chans=128, conv_op=nn.Conv2d):
        super(Decoder_UP2, self).__init__()

        self.conv1x1 = single_conv_relu_batch(in_chans*2, out_chans, kernel_size=1, stride=1, padding=0,
                                              conv_op=conv_op)
        self.up = double_conv_relu_batch(in_chans+out_chans, out_chans, kernel_size=3, padding=1,  conv_op=conv_op)

    def forward(self, f2, f3):
        y = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)

        f3 = self.conv1x1(f3)
        f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([y, f3], dim=1) # 64 64 32 16
        f1 = self.up(x)
        return f1

class Decoder_UP3(nn.Module):
    def __init__(self, in_chans=128, out_chans=64,  conv_op=nn.Conv2d):
        super(Decoder_UP3, self).__init__()

        self.conv1x1 = single_conv_relu_batch(in_chans*2, out_chans, kernel_size=1, stride=1, padding=0, conv_op=conv_op)

        self.up = double_conv_relu_batch(in_chans*2, out_chans, kernel_size=3, padding=1, conv_op=conv_op)

    def forward(self, x, f1, f2):
        f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)

        f2 = self.conv1x1(f2)
        f2= F.interpolate(f2, scale_factor=4, mode='bilinear', align_corners=True)
        x = torch.cat([x, f1, f2], dim=1) # 32  32 16 8 4

        f0 = self.up(x)
        return f0

class InstanceCounter(nn.Module):
    def __init__(self, input_n_filters, out_chs=16, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters  #将输入特征图的通道数保存为实例变量
        self.n_filters = out_chs
        self.out_filter = out_chs
        self.usegpu = usegpu
        self.output = nn.Sequential(nn.Linear(self.out_filter, 1),
                                    nn.Sigmoid())
        self.cnn = nn.Sequential(
                                 nn.MaxPool2d(2,2),
                                 single_conv_relu_batch(in_ch=self.input_n_filters, out_ch=self.out_filter, kernel_size=3, stride=1, padding=1),
                                 single_conv_relu_batch(in_ch=self.out_filter, out_ch=self.out_filter, kernel_size=3, stride=1, padding=1),
                                 nn.AdaptiveAvgPool2d((1, 1))
                                 )   #从输入特征图中提取特征

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)
        x = self.output(x)
        return x

class RsUnet01(nn.Module):
    def __init__(self,  in_chans=[64, 128, 256, 512, 1024], out_chs=20,  out_class=2):
        super(RsUnet01, self).__init__()
        self.down = Resnet18(in_channels=3, out_channels=in_chans)   #下采样

        self.ins_counter = InstanceCounter(in_chans[0], out_chs=out_chs)   #计算实例数量
        self.ins_segHead = instanceSegmenationHead(in_chans[0], out_chans=out_chs)
        self.sem_segHead = instanceSegmenationHead(in_chans[0], out_class)


        self.up0 = Decoder_UP0(in_chans[4], in_chans[3])
        self.up1 = Decoder_UP1(in_chans[3], in_chans[2])
        self.up2 = Decoder_UP2(in_chans[2], in_chans[1])
        self.up3 = Decoder_UP3(in_chans[1], in_chans[0])

        self.rc_skip = RC_skip(in_chans[2], in_chans[1])

        self.apply(self._init_weight)

    def forward(self, x):

        features, x = self.down(x)   #接受输入x并返回下采样后的特征features和剩余的部分x

        inter = []      #存储下采样和上采样之间的中间特征
        inter.append(features[0])    #将下采样得到的第一个特征添加到 inter列表中，通常是编码器中提取的最底层的特征

        s = self.rc_skip(features[3], features[2], features[1], features[0])
        inter.append(s)

        f4 = x   #将剩余的部分x赋值给 f4
        f3 = self.up0(inter[0], x)
        f2 = self.up1(f3, f4)
        f1 = self.up2(f2, f3)
        f0 = self.up3(inter[1], f1, f2)


        ins_pre = self.ins_segHead(f0)
        seg_pre = self.sem_segHead(f0)
        n_instance = self.ins_counter(f0)

        #return  seg_pre
        return(seg_pre,ins_pre,n_instance)
                 #ins_pre, n_instance)

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



