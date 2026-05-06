from torch import nn
import torch
import torchvision.models as models
import torch.nn.functional as F


# 定义解码器中的卷积块
class expansive_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(expansive_block, self).__init__()

        # 卷积块的结构
        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, d, e=None):
        # 上采样
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接
        if e is not None:
            cat = torch.cat([e, d], dim=1)
            out = self.block(cat)
        else:
            out = self.block(d)
        return out

# 定义最后一层卷积块
def final_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
    )
    return block



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


# 定义 Resnet34_Unet 类
class Resnet34_Unet(nn.Module):
    # 定义初始化函数
    def __init__(self, in_channel, out_channel, pretrained=False):
        # 调用 nn.Module 的初始化函数
        super(Resnet34_Unet, self).__init__()
        
        # 创建 ResNet34 模型
        self.resnet = models.resnet34(pretrained=pretrained)
        # 定义 layer0，包括 ResNet34 的第一层卷积、批归一化、ReLU 和最大池化层
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )

        # 定义 Encode 部分，包括 ResNet34 的 layer1、layer2、layer3 和 layer4
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # 定义 Bottleneck 部分，包括两个卷积层、ReLU、批归一化和最大池化层
        self.bottleneck = torch.nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # 定义 Decode 部分，包括四个 expansive_block 和一个 final_block
        self.conv_decode4 = expansive_block(1024+512, 512, 512)
        self.conv_decode3 = expansive_block(512+256, 256, 256)
        self.conv_decode2 = expansive_block(256+128, 128, 128)
        self.conv_decode1 = expansive_block(128+64, 64, 64)
        self.conv_decode0 = expansive_block(64, 32, 32)
        self.conv_decode = expansive_block(32, 24, 24)
        self.final_layer = feHead(24, out_channel)

    # 定义前向传播函数
    def forward(self, x):
        # 执行 layer0
        x = self.layer0(x)
        # 执行 Encode
        encode_block1 = self.layer1(x)
        encode_block2 = self.layer2(encode_block1)
        encode_block3 = self.layer3(encode_block2)
        encode_block4 = self.layer4(encode_block3)

        # 执行 Bottleneck
        bottleneck = self.bottleneck(encode_block4)

        # 执行 Decode
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        decode_block0 = self.conv_decode0(decode_block1)
        decode = self.conv_decode(decode_block0)
        seg_head, ins_head, count = self.final_layer(decode)

        return seg_head, ins_head, count

#
# flag = 0
#
# if flag:
#     image = torch.rand(1, 3, 572, 572)
#     Resnet34_Unet = Resnet34_Unet(in_channel=3, out_channel=1)
#     mask = Resnet34_Unet(image)
#     print(mask.shape)
#
# # 测试网络
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Resnet34_Unet(in_channel=1, out_channel=1, pretrained=True).to(device)

