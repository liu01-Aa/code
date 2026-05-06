import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
# from timm.layers import trunc_normal_
import torch.nn.functional as F
import math
from typing import Iterable
import torch.fft as fft
from torch.nn import Softmax, LayerNorm
from torch.nn.parameter import Parameter

#   singleConv模块封装了一个简单的卷积块
class singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):  # 卷积操作的类型——二维卷积
        super(singleConv, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),  # 对输入特征图进行卷积操作，提取特征。
            nn.BatchNorm2d(out_ch),  # 对卷积层的输出进行归一化处理，加速模型的训练过程，提高模型的稳定性。
            nn.ReLU(True)  # 创建一个激活函数层，对批量归一化的输出进行非线性变换，引入非线性特性，增强模型的表达能力。
        )

    def forward(self, x):
        x = self.conv(x)  # x——输入数据 经过处理后返回
        return x


class singleConv0(nn.Module): # x 依次通过卷积层和批归一化层，输出结果。
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
        super(singleConv0, self).__init__()
        self.conv = nn.Sequential(
            conv_op(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class single_conv_relu_batch(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
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
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
        super(double_conv_relu_batch, self).__init__()
        self.conv = nn.Sequential(
            single_conv_relu_batch(in_ch, out_ch, kernel_size=1, stride=1, padding=0, conv_op=conv_op),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv_op(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)  # 进行两次连续的卷积 - 批量归一化 - ReLU 操作。这意味着输入数据会经过更多层的处理，特征能够得到更充分的提取和转换。
        )  # 进行了两次 ReLU 激活操作，能够引入更强的非线性，使模型可以学习到更复杂的函数映射关系，从而提高模型的表达能力。

    def forward(self, x):
        x = self.conv(x)
        return x


class doubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d):
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


# 通道注意力机制 让模型自动学习每个通道的重要性，从而为不同通道分配不同的权重，提升模型对特征的表达能力。
# ratio：降维系数，默认值为 16，用于控制多层感知机（MLP）中间层的通道数，起到降维和减少计算量的作用。

class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):  # 初始化模块的参数
        super(channel_attention, self).__init__()
        # 平均池化 将输入特征图的每个通道自适应地池化到大小为 1x1 的特征图。通过平均池化，可以获取每个通道的全局平均信息。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化 全局最大信息。
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 多层感知机MLP定义  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # bias=False 表示不使用偏置项
        self.relu1 = nn.ReLU()  # 在 fc1 卷积层之后引入非线性，
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()  # 定义一个 Sigmoid 激活函数层，用于将输出的特征图的每个通道的值映射到 [0, 1] 区间，作为每个通道的注意力权重。

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out  # 将平均池化和最大池化的输出相加，综合考虑两种池化方式得到的信息，获取输入特征图的全局信息。
        return self.sigmoid(out)  # 得到每个通道的注意力权重，取值范围在 [0, 1] 之间。


# 空间注意力机制 让模型关注输入特征图中不同空间位置的重要性，从而为不同的空间位置分配不同的权重。
# 这些权重可以用于后续的特征图加权操作，以突出重要的空间位置，抑制不重要的空间位置。
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()
        # 声明卷积核为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 进行相应的same padding填充（根据卷积核大小的不同，通过计算得到合适的填充大小（p=K-1/2），就可以在步长为 1 的卷积操作中实现 “same padding”
        # 保证特征图的空间尺寸不变，即输出特征图的空间尺寸与输入特征图的空间尺寸相同。）
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # 将卷积层的输出映射到 [0, 1] 区间，作为空间注意力权重。

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 平均池化 得到每个空间位置的平均特征值 x——输入特征图
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化 得到每个空间位置的最大特征值
        # 拼接操作
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 7x7卷积填充为3，输入通道为2，输出通道为1
        return self.sigmoid(x)


class CBAM(nn.Module):  # 结合了通道注意力机制和空间注意力机制 两者结合可以让模型更加聚焦于重要的通道和空间位置，提高模型的性能。
    def __init__(self, in_chs, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channel_attn = channel_attention(in_chs=in_chs, ration=ratio)
        self.spatial_attn = spatial_attention(kernel_size=kernel_size)

    def forward(self, inputs):
        x = self.channel_attn(inputs)  # 输入特征图 inputs
        x = self.spatial_attn(x)

        return x  # 返回经过通道注意力机制和空间注意力机制处理后的特征图 x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))  # sigmoid为输入特征图的每个空间位置分配了一个权重值。
        return out * x  # 将生成的空间注意力图 out 与输入特征图 x 逐元素相乘。这样做可以增强特征图中注意力权重较大的空间位置
        # 抑制权重较小的位置，最终返回经过空间注意力机制处理后的特征图。 通过这种方式来调整输入特征，使模型能够更聚焦于重要的空间区域。


# nn.Linear定义一个全连接层（也称作线性层）。输入特征维度为 patch_size * patch_size，输出特征维度为 output_dim // 2。这个全连接层用于处理局部区域的特征。
# 定义了一个可学习的参数 prompt，其初始值是从标准正态分布中随机采样得到的。在模型训练过程中，prompt 的值会根据梯度进行更新，以帮助模型更好地完成局部特征的筛选和注意力计算等任务。
# 定义了一个可学习的参数 top_down_transform，其初始值为单位矩阵。在模型训练过程中，top_down_transform 的值会根据梯度进行更新，用于对筛选后的局部特征进行线性变换，帮助模型更好地处理和整合局部信息。
# torch.randn(output_dim)生成一个张量 有output_dim个元素
# torch.eye(output_dim) 创建一个形状为 (output_dim, output_dim) 的单位矩阵。
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):  # 局部区域（patch）的大小。
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)  # 用于对self.mlp1的输出（对其最后一个维度）进行归一化处理，加速模型的训练和提高模型的稳定性。
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # 改变张量（Tensor）的维度顺序 张量 x 的形状会从 (B, C, H, W) 变为 (B, H, W, C)。
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P,
                                                 P)  # (B, H/P, W/P, P, P, C) 其中 B 是批量大小，H/P 和 W/P 分别是高度和宽度方向上的窗口数量，P 是窗口的高度和宽度，C 是通道数。
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)
        # 这三行代码的整体作用是将输入的特征图 x 划分为多个不重叠的局部区域，对这些局部区域进行形状重塑，然后对每个局部区域的通道信息进行压缩，以便后续进行局部特征的处理和分析。
        # 最终得到的 local_patches 张量形状为 (B, H/P*W/P, P*P)，其中 B 是批量大小，H/P*W/P 是总的窗口数量，P*P 是每个窗口内的元素数量。
        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim) 得到将局部区域在不同特征维度上的注意力权重。
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim) 让模型更加关注每个局部区域中重要的特征信息。

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform
        # 这四行代码实现了一个特征筛选和线性变换的过程。首先计算 local_out 与 self.prompt 之间的余弦相似度，得到一个相似度矩阵；
        # 然后将相似度矩阵裁剪为掩码，用于筛选重要的局部特征；最后对筛选后的局部特征进行线性变换，以学习更复杂的特征表示。
        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output
        # 对处理后的局部特征进行形状恢复、维度调整、上采样和卷积操作，最终得到与输入特征图具有相同空间尺寸且经过进一步特征提取的输出特征图。
        # 这个过程有助于将局部特征的信息整合并恢复到原始图像的空间尺度，以便后续用于图像分类、目标检测等任务。最后通过 return output 返回最终的输出结果。


# ECA高效的通道注意力机制 强特征图中重要通道的特征表达，抑制不重要通道的特征，提升模型对特征的利用效率，进而提高模型的性能。
class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)  # 进行自适应平均池化操作，将每个通道的特征信息全局平均。 b c h w高宽变为1
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)  # (B, C, 1, 1)
        out = out.view(x.size(0), 1, x.size(1))  # (B, 1, C) 转换为适合一维卷积层输入的形状
        out = self.conv(out)  # (B, 1, C)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x

class MSFE(nn.Module):
    def __init__(self, in_ch,out_ch):
        super().__init__()
        # 水平方向多尺度卷积 (1x3 kernel)
        self.conv_h1 = nn.Conv2d(in_ch, in_ch, (1, 3), padding=(0, 1), dilation=1, groups=in_ch)
        self.conv_h2 = nn.Conv2d(in_ch, in_ch, (1, 3), padding=(0, 2), dilation=2, groups=in_ch)
        # 垂直方向多尺度卷积 (3x1 kernel)
        self.conv_v1 = nn.Conv2d(in_ch, in_ch, (3, 1), padding=(1, 0), dilation=1, groups=in_ch)
        self.conv_v2 = nn.Conv2d(in_ch, in_ch, (3, 1), padding=(2, 0), dilation=2, groups=in_ch)
        # 融合层
        self.fuse = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        h1 = self.conv_h1(x)  # 水平-尺度1
        h2 = self.conv_h2(x)  # 水平-尺度2
        v1 = self.conv_v1(x)  # 垂直-尺度1
        v2 = self.conv_v2(x)  # 垂直-尺度2
        y=self.fuse(h1 + h2 + v1 + v2) + x
        y = self.maxpool(y)  # MSFE修改 x---y
        y = self.conv1(y)
        return y # 多尺度融合 + 残差连接

class DepthwiseConv1dxDx1WithDilation(nn.Module):
    def __init__(self, input_channels, kernel_size, dilation=1):
        super(DepthwiseConv1dxDx1WithDilation, self).__init__()
        if dilation == 1:
            # 1xd的深度卷积核
            self.depthwise_conv_1xd = nn.Conv2d(input_channels, input_channels, kernel_size=(1, kernel_size),
                                                padding=(0, kernel_size // 2), groups=input_channels,
                                                dilation=(1, dilation))
            # dx1的深度卷积核
            self.depthwise_conv_dx1 = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, 1),
                                                padding=(kernel_size // 2, 0), groups=input_channels,
                                                dilation=(dilation, 1))
        else:
            # 1xd的深度卷积核
            self.depthwise_conv_1xd = nn.Conv2d(input_channels, input_channels, kernel_size=(1, kernel_size),
                                                padding=(0, dilation), groups=input_channels,
                                                dilation=(1, dilation))
            # dx1的深度卷积核
            self.depthwise_conv_dx1 = nn.Conv2d(input_channels, input_channels, kernel_size=(kernel_size, 1),
                                                padding=(dilation, 0), groups=input_channels,
                                                dilation=(dilation, 1))

    def forward(self, x):

        x = self.depthwise_conv_1xd(x)
        x = self.depthwise_conv_dx1(x)
        return x



class ConvBN(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation=1):
        super(ConvBN, self).__init__()
        self.conv = DepthwiseConv1dxDx1WithDilation(in_channels, kernel_size, dilation)
        # self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x

class GAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(GAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3, padding=1,
                                    groups=in_dim // 8)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=3, padding=1,
                                  groups=in_dim // 8)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        # self.dwc = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=(3, 3), padding=1, groups=in_dim)
        self.dilated_conv2 = ConvBN(in_dim, kernel_size=3, dilation=1)
        self.dilated_conv3 = ConvBN(in_dim, kernel_size=3, dilation=2)
        self.dilated_conv4 = ConvBN(in_dim, kernel_size=3, dilation=3)

        self.conv1x1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.norm = LayerNorm(in_dim, eps=1e-6)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        features = self.gamma * out + x
        features1 = features.permute(0, 2, 3, 1)
        features1 = self.norm(features1)
        features1 = features1.permute(0, 3, 1, 2)
        dilated2 = self.dilated_conv2(features1)
        dilated2 = F.gelu(dilated2)
        dilated3 = self.dilated_conv3(features1)
        dilated3 = F.gelu(dilated3)
        dilated4 = self.dilated_conv4(features1)
        dilated4 = F.gelu(dilated4)
        output = dilated2 + dilated3 + dilated4
        output = self.conv1x1(output)
        output = F.gelu(output)
        out = output + features
        return out


#  out * x将通道重要性权重与输入特征图进行逐通道的乘法操作，对每个通道的特征进行加权，增强重要通道的特征表达，抑制不重要通道的特征，最后返回加权后的特征图。
# cross_conv自定义的卷积模块，其主要作用是对输入特征图进行一系列的卷积、下采样、注意力机制应用等操作，以提取多尺度的特征信息，并通过注意力机制增强特征表达，最后将处理后的特征进行融合和转换，得到输出特征图，
class cross_conv(nn.Module):  # 编码器模块
    def __init__(self, in_ch, out_ch, conv_op=nn.Conv2d):
        super(cross_conv, self).__init__()
        #32, 64, 128, 256
        self.conv0 = singleConv0(in_ch, out_ch, kernel_size=1, stride=2, padding=0, conv_op=nn.Conv2d)  # 空间尺寸减半 应用-跳跃连接 降维

        # 2. 替换部分：用 MSFE 替换所有 1x3 和 3x3 conv  修改-替换MSFE
        self.msfe = MSFE(in_ch, out_ch)  # 输入通道需与 in_ch 一致（如64）
        #修改-替换MSFE--------------------------------------------

        # self.maxpool = nn.MaxPool2d(2, 2)

        # self.conv1 = nn.Sequential(singleConv(in_ch, out_ch, kernel_size=3, stride=1, padding=1, conv_op=nn.Conv2d),
        #                            singleConv(out_ch, out_ch, kernel_size=3, stride=1, padding=1,
        #                                       conv_op=nn.Conv2d))  # 尺寸不变 改变通道数 应用-主干特征提取 MSFE注释
        # 3. 主干特征路径（简化版，仅保留必要操作）MSFE--添加
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.conv3 = singleConv(in_ch, out_ch, kernel_size=2, stride=2, padding=0, conv_op=conv_op)
        self.conv3_1 = singleConv(out_ch, out_ch, kernel_size=(1, 3), stride=1, padding=(0, 1), conv_op=conv_op)  # 水平
        self.conv3_2 = singleConv(out_ch, out_ch, kernel_size=(3, 1), stride=1, padding=(1, 0), conv_op=conv_op)  # 垂直

        self.sa = SpatialAttentionModule()
        self.eca = ECA(in_channel=out_ch)

        self.bn = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(0.1)
        # 正则化技术，它的主要目的是防止神经网络过拟合。在训练过程中，
        # 该层会以一定的概率随机 “丢弃”（置为 0）输入特征图中的某些通道，通过随机丢弃一些通道，可以减少特征之间的依赖关系，
        # 避免模型过度依赖某些特定的通道，从而提高模型的泛化能力。以此增加模型的泛化能力。
        self.relu = nn.ReLU()  # ReLU 函数引入了非线性因素 图像的特征往往是高度非线性的，ReLU 可以帮助网络更好地捕捉这些特征。

        self.conv_block1 = nn.Sequential(nn.Conv2d(out_ch * 2, in_ch, kernel_size=1, padding=0),
                                         nn.BatchNorm2d(in_ch),
                                         nn.ReLU(inplace=True)) #MSFE修改 out_ch * 3---out_ch * 2

        self.lga2 = LocalGlobalAttention(output_dim=out_ch, patch_size=2)
        self.lga4 = LocalGlobalAttention(output_dim=out_ch, patch_size=4)

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(out_ch))

        self.conv_block3 = nn.Sequential(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                                         nn.BatchNorm2d(out_ch)
                                         )
        self.conv1x1 = singleConv0(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)#MSFE修改 out_ch * 3---out_ch * 2
        self.relu = nn.ReLU(inplace=True)
        self.gam = GAM_Module(in_ch)
        # 在 __init__ 中添加调整通道的卷积层
        self.adjust_gam = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # 1x1卷积不改变空间尺寸


    def forward(self, x):
        g = self.conv0(x)  #[16, out_ch, 160, 80] 特征提取的初始层 用来作残差连接
        y = self.msfe(x)  # MSFE添加
        # y = self.maxpool(y) #MSFE修改 x---y
        # y = self.conv1(y) #修改----maxpool conv1加进msfe
        # y = self.conv1(y)  #[16, out_ch, 160, 80] MSFE注释--两个3*3 conv

        # k = self.conv3(x) # MSFE注释--2*2 conv
        # k1 = self.conv3_1(k)  #[16, out_ch, 160, 80] # MSFE注释--1*3 conv
        # k2 = self.conv3_2(k)  #[16, out_ch, 160, 80] # MSFE注释--1*3 conv


        # x = torch.cat([y, k1, k2], dim=1)  #[16, out_ch*3, 160, 80] 把水平 垂直特征图和最大池后卷积的特征图拼接 MSFE注释
        #x = self.channnel_attn(x) * x        
        x = torch.cat([g, y], dim=1)  # 拼接不同路径特征 # MSFE添加

        # f=x
        # f = self.conv1x1(g)

        x = self.conv_block1(x) # 1*1 conv
        x_lga2 = self.lga2(x)
        x_lga4 = self.lga4(x)
        # gam = self.gam(x)
        x = self.conv_block2(x)
        # print(f"gam shape: {gam.shape}, x_lga4 shape: {x_lga4.shape}, x shape: {x.shape}")
        # 根据输出调整模块或插入维度转换操作
        # 在 forward 中修改为：
        # gam_adjusted = self.adjust_gam(gam)  # [4, 128, 128, 128]
        # print(f"gam shape: {gam.shape}, x_lga4 shape: {x_lga4.shape}, x shape: {x.shape}")
        # 根据输出调整模块或插入维度转换操作
        # print(f"gam shape: {gam.shape}, x_lga4 shape: {x_lga4.shape}, x shape: {x.shape}")
        x = x_lga2 + x_lga4 + x  # 经过两个不同尺寸的局部注意力模块（得到额外的特征表示）和卷积后的x相加
        # 在输入x的基础上叠加了额外学习到的特征信息，符合残差连接的思想。



        x = self.conv_block3(x)
        x = self.eca(x)  # ECA 通道注意力
        x = self.sa(x)  # 空间注意力
        x = self.drop(x)  # 正则化
        x = self.bn(x)  # 归一化
        x = self.relu(x)  #引入非线性 以上这些操作使得 x 学习到了丰富的特征信息

        # x = self.relu(x + f)  #逐元素相加使得每个位置的元素综合了x和f两个特征图在该位置的信息，实现了特征的融合
        # 残差连接后 使用ReLU激活函数通过引入非线性、缓解梯度消失问题
        x = self.relu(x + g)
        # x = self.relu(self.bn(x + f + g))
        return x

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # 通道维度放到最后
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # 通道维度放到最前
        x1 = self.relu(x)
        x = self.pool(x1)
        return x, x1

class Resnet18(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, conv_op=nn.Conv2d):
        super(Resnet18, self).__init__()
        #[64, 128, 256, 512,1024]
        self.conv = singleConv(in_channels, out_channels[0], kernel_size=kernel_size, stride=stride,
                               padding=padding, conv_op=conv_op)
        # 修改2------把conv改为stem
        # self.stemk = Stem(in_channels, out_channels=out_channels[0])

        self.cross_0 = cross_conv(out_channels[0], out_channels[1], conv_op=conv_op)
        self.cross_1 = cross_conv(out_channels[1], out_channels[2], conv_op=conv_op)
        self.cross_2 = cross_conv(out_channels[2], out_channels[3], conv_op=conv_op)
        self.cross_3 = cross_conv(out_channels[3], out_channels[4], conv_op=conv_op)

    def forward(self, x):
        feature = []
        x = self.conv(x)  # F0输入3输出64

        # x, k0 = self.stemk(x) # 修改2------把self.conv(x)改为self.stemk(x)
        # print(f"Stem output x shape: {x.shape}")
        feature.append(x)  #

        x = self.cross_0(x)  # F1输入64输出128
        feature.append(x)

        x = self.cross_1(x)
        feature.append(x)

        x = self.cross_2(x)
        feature.append(x)

        x = self.cross_3(x)  # F 0 1 2 3, 4
        return feature[::-1], x  #将存储有每个阶段特征的列表 feature逆序返回（F3 F2 F1 F0），并同时返回最后一阶段的输出x(F4)


class instanceSegmenationHead(nn.Module):  # 调整特征图的通道数
    def __init__(self, in_chans=54, out_chans=36):
        super(instanceSegmenationHead, self).__init__()
        self.ins_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0)

    def forward(self, y):
        x = self.ins_conv(y)
        return x


# 是对输入的来自网络不同深度的四个特征图 d1、d2、d3、d4 进行处理。最终输出六个经过处理和融合的特征图 d3_1、d2_1、d2_2、d1_1、d1_2、d1_3，用于后续的特征提取和预测任务。
class RC_skip(nn.Module):
    def __init__(self, in_chans=256, out_chans=128, conv_op=nn.Conv2d):
        super(RC_skip, self).__init__()
        self.convtrans_d3 = nn.ConvTranspose2d(in_chans * 2, out_chans * 2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d2_1 = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, padding=0)
        self.convtrans_d2_2 = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, padding=0)

        self.convtrans_d1_1 = nn.ConvTranspose2d(out_chans, out_chans // 2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d1_2 = nn.ConvTranspose2d(out_chans, out_chans // 2, kernel_size=2, stride=2, padding=0)
        self.convtrans_d1_3 = nn.ConvTranspose2d(out_chans, out_chans // 2, kernel_size=2, stride=2, padding=0)
        # 用转置卷积层（nn.ConvTranspose2d）实现了上采样操作 增大特征图的尺寸
        self.up_d1_1 = single_conv_relu_batch(out_chans, out_chans // 2, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d1_2 = single_conv_relu_batch(out_chans, out_chans // 2, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d1_3 = single_conv_relu_batch(out_chans, out_chans // 2, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d2_1 = single_conv_relu_batch(in_chans, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d2_2 = single_conv_relu_batch(in_chans, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
        self.up_d3 = single_conv_relu_batch(in_chans * 2, in_chans, kernel_size=3, padding=1, conv_op=conv_op)
        # self.conv1x1 = single_conv_relu_batch(in_chans, out_chans//2, kernel_size=1, stride=1, padding=0, conv_op=conv_op)

    def forward(self, d1, d2, d3, d4):  # F0 F1 F2 F3网络深度节点输出
        d4 = self.convtrans_d3(d4) # d4扩大尺寸
        if d3.size(2) != d4.size(2) or d3.size(3) != d4.size(3):
            d3 = F.interpolate(d3, size=(d4.size(2), d4.size(3)), mode='bilinear', align_corners=True)

        d3_1 = torch.cat([d3, d4], dim=1)  # F2 F3 融合 浅层特征映射和经过转置卷积得到的深层特征映射融合
        d3_1 = self.up_d3(d3_1)  # 对拼接后的特征信息进行有效整合

        d3 = self.convtrans_d2_1(d3)
        if d3.size(2) != d2.size(2) or d3.size(3) != d2.size(3):
            d3 = F.interpolate(d3, size=(d2.size(2), d2.size(3)), mode='bilinear', align_corners=True)

        d2_1 = torch.cat([d2, d3], dim=1)  # F1 F2融合
        d2_1 = self.up_d2_1(d2_1)
        # 主要操作：对不同深度的特征图进行上采样、尺寸匹配、特征拼接以及进一步的特征处理，
        # 从而整合多尺度的特征信息，最终输出多个经过处理和融合的特征图
        d2_2 = self.convtrans_d2_2(d3_1)
        if d3.size(2) != d2_2.size(2) or d3.size(3) != d2_2.size(3):
            d3 = F.interpolate(d3, size=(d2_2.size(2), d2_2.size(3)), mode='bilinear', align_corners=True)

        d2_2 = torch.cat([d2, d2_2], dim=1)  # 512
        d2_2 = self.up_d2_2(d2_2)

        d1_1 = self.convtrans_d1_1(d2)
        #  修改——F.interpolate 调整它们的尺寸，使其与 d1 的高度和宽度一致。
        # if d1_1.size(2) != d1.size(2) or d1_1.size(3) != d1.size(3):
        #     d1_1 = F.interpolate(d1_1, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)

        d1_1 = torch.cat([d1, d1_1], dim=1)
        d1_1 = self.up_d1_1(d1_1)

        d1_2 = self.convtrans_d1_2(d2_1)
        #  修改——F.interpolate 调整它们的尺寸，使其与 d1 的高度和宽度一致。
        # if d1_2.size(2) != d1.size(2) or d1_2.size(3) != d1.size(3):
        #     d1_2 = F.interpolate(d1_2, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)
        #  修改——打印看尺寸是否相同
        # print("d1 shape:", d1.shape)
        # print("d1_2 shape:", d1_2.shape)

        d1_2 = torch.cat([d1, d1_2], dim=1)
        d1_2 = self.up_d1_2(d1_2)

        d1_3 = self.convtrans_d1_3(d2_2)
        #  修改——F.interpolate 调整它们的尺寸，使其与 d1 的高度和宽度一致。
        # if d1_3.size(2) != d1.size(2) or d1_3.size(3) != d1.size(3):
        #     d1_3 = F.interpolate(d1_3, size=(d1.size(2), d1.size(3)), mode='bilinear', align_corners=True)

        d1_3 = torch.cat([d1, d1_3], dim=1)
        d1_3 = self.up_d1_3(d1_3)

        # x = torch.cat([d1, d1_1, d1_2, d1_3], dim=1)
        # x = self.conv1x1(x)

        return d3_1, d2_1, d2_2, d1_1, d1_2, d1_3
# 新增SelfAttentionBlock
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1)  # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()  # (B, h*w, C)

        sim_map = torch.matmul(query, key)

        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # (B, h*w, K)

        context = torch.matmul(sim_map, value)  # (B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # (B, C, h, w)

        context = self.out_project(context)  # (B, C, h, w)
        return context

    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]


# 新增conv_1x1
def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

# 新增SFF
class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel * 2, in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels=in_channel,
            transform_channels=in_channel // 2,
            out_channels=in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )

    def forward(self, x_small, x_big):
        img_size = x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big) # x是64通道
        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out

# class Decoder_UP0(nn.Module):
#     def __init__(self, in_chans=1024, out_chans=512, conv_op=nn.Conv2d):
#         super(Decoder_UP0, self).__init__()
#         self.up0 = double_conv_relu_batch(in_chans + out_chans, out_chans, kernel_size=3, padding=1,
#                                           conv_op=conv_op)
#     def forward(self, x, f4): #  inter[0]和f4
#         y = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
#         x = torch.cat([x, y], dim=1)  # 512+1024  inter[0]和f4
#         f3 = self.up0(x)  # 512 对拼接的特征图进行卷积得f3
#         return f3
class Decoder_UP(nn.Module):
    def __init__(self, in_chans, out_chans, conv_op=nn.Conv2d): # 1024 512
        super(Decoder_UP, self).__init__()
        self.conv_small = conv_1x1(in_chans, in_chans)
        self.conv_big = conv_1x1(out_chans, out_chans)
        self.conv_pj = conv_1x1(in_chans+out_chans, out_chans)
        # self.up0 = double_conv_relu_batch( out_chans*2, out_chans, kernel_size=3, padding=1,
        #                                   conv_op=conv_op) # 注释
        # # 用DenseDecoderBlock替换原来的up0
        # self.dense_block = DenseDecoderBlock(out_chans * 2, growth_rate=32)
        #
        # # 最终的投影层
        # self.final_conv = nn.Sequential(
        #     nn.Conv2d(out_chans * 2 + 32 * 4, out_chans, 1),  # 计算输入通道数: out_chans*2 + n_layers*growth_rate
        #     nn.BatchNorm2d(out_chans),
        #     nn.ReLU(inplace=True)
        # ) 之前只对解码器1进行加强时

        # 修正1：明确输入通道数
        self.dense_in_ch = out_chans * 2  # 1024 (512+512)

        # 修正2：统一growth_rate为32
        self.growth_rate = 32
        self.n_layers = 4

        self.dense_block = DenseDecoderBlock(
            in_ch=self.dense_in_ch,
            growth_rate=self.growth_rate,
            n_layers=self.n_layers
        )

        # 修正3：准确计算输出通道
        dense_out_ch = self.dense_in_ch + self.growth_rate * self.n_layers

        self.final_conv = nn.Sequential(
            nn.Conv2d(dense_out_ch, out_chans, 1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )

        self.attention = SelfAttentionBlock(
            key_in_channels=out_chans,
            query_in_channels=out_chans,
            transform_channels=out_chans // 2,
            out_channels=out_chans,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
    def forward(self, x, f4): #  inter[0]和f4 inter[1]和f3
        # print(f"x channels: {x.shape}")
        y = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        y=self.conv_small(y)
        x= self.conv_big(x)
        # print(f"x channels: {x.shape}, y channels: {y.shape}") #512 40 20/1024 40 20
        pj = torch.cat([x, y], dim=1)  # 512+1024  inter[0]和f4  512+256 inter[1]和f3 pj的通道数是1536 40 20
        pj =self.conv_pj(pj) # 512 40 20
        new_x = self.attention(pj,pj,x)
        # print(f"new_x channels: {new_x.shape}") # new_x channels: 512
        # f3 = torch.cat([new_x, x], dim=1) # 512 512 f3 1024 注释
        # f3 = self.up0(f3)  # 512 对拼接的特征图进行卷积得f3 变为512 注释
        # print(f"f3 channels: {f3.shape}")
        # 使用密集连接块
        dense_input = torch.cat([new_x, x], dim=1)
        # print(f"Dense输入形状: {dense_input.shape}")  # 应为[batch, dense_in_ch, h, w]
        dense_out = self.dense_block(dense_input)
        # print(f"Dense输出形状: {dense_out.shape}")  # 应为[batch, dense_in_ch+growth_rate*n_layers, h, w]

        f3 = self.final_conv(dense_out)
        return f3 # 一系列操作加强后的inter0


class Decoder_UP3(nn.Module):
    def __init__(self, in_chans=128, out_chans=64, conv_op=nn.Conv2d):
        super(Decoder_UP3, self).__init__()
        # 保持与Decoder_UP类似的结构
        self.conv_small = conv_1x1(in_chans, in_chans)  # 用于处理f1
        self.conv_big = conv_1x1(out_chans, out_chans)  # 用于处理inter[6]
        self.conv_pj = conv_1x1(in_chans + out_chans + out_chans, out_chans)  # 拼接后的投影
        self.conv1x1 = single_conv_relu_batch(in_chans * 2, out_chans, kernel_size=1, stride=1, padding=0,
                                              conv_op=conv_op)  # f4变为256

        # 使用ECA注意力替代Self-Attention
        self.attention = ECA(out_chans)  # 轻量级注意力

        # 最终的上采样卷积
        # self.up0 = double_conv_relu_batch(out_chans * 2, out_chans,
        #                                   kernel_size=3, padding=1, conv_op=conv_op) 注释
        # 修改输入通道计算

        # 修正通道计算
        self.dense_in_ch = out_chans * 2  # 128
        self.growth_rate = 8  # 低层的growth_rate
        self.n_layers = 2

        self.dense_block = DenseDecoderBlock(
            in_ch=self.dense_in_ch,
            growth_rate=self.growth_rate,
            n_layers=self.n_layers
        )

        dense_out_ch = self.dense_in_ch + self.growth_rate * self.n_layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(dense_out_ch, out_chans, 1),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, f1, f):
        """
        x: inter[6] (来自RC_skip的输出)
        f1: 上一层Decoder的输出 (f1)
        """
        # 上采样f1以匹配x的空间尺寸
        y = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
        y = self.conv_small(y)  # [B, in_chans, H, W]
        x = self.conv_big(x)  # [B, out_chans, H, W]
        f = self.conv1x1(f)  # f4 1024变为256

        f = F.interpolate(f, scale_factor=4, mode='bilinear',
                          align_corners=True)  # 对 f4进行上采样，将其尺寸增加四倍，inter[1], f3, f4尺寸相同
        # 拼接并投影
        pj = torch.cat([x, y, f], dim=1)  # [B, in_chans+out_chans, H, W]
        pj = self.conv_pj(pj)  # [B, out_chans, H, W]

        # 应用轻量级注意力
        new_x = self.attention(pj) * pj  # ECA是通道注意力，直接相乘

        # 最终拼接和卷积
        # f0 = torch.cat([new_x, x], dim=1)  # [B, out_chans*2, H, W]
        # f0 = self.up0(f0)  # [B, out_chans, H, W]
        # 使用密集连接块
        dense_input = torch.cat([new_x, x], dim=1)
        dense_out = self.dense_block(dense_input)

        f0 = self.final_conv(dense_out)
        # print(f"f0 channels: {f0.shape}")
        return f0
class Decoder_UP1(nn.Module):
        def __init__(self, in_chans, out_chans, conv_op=nn.Conv2d):  # 512 256
            super(Decoder_UP1, self).__init__()
            self.conv_small = conv_1x1(in_chans, in_chans)
            self.conv_big = conv_1x1(out_chans, out_chans)
            self.conv_pj = conv_1x1(in_chans + out_chans+ out_chans, out_chans)
            # self.up0 = double_conv_relu_batch(out_chans * 2, out_chans, kernel_size=3, padding=1,
            #                                   conv_op=conv_op)
            self.conv1x1 = single_conv_relu_batch(in_chans * 2, out_chans, kernel_size=1, stride=1, padding=0,
                                                          conv_op=conv_op) # f4变为256
            # 修正通道计算
            self.dense_in_ch = out_chans * 2  # 512
            self.growth_rate = 16  # 中层的growth_rate
            self.n_layers = 3

            self.dense_block = DenseDecoderBlock(
                in_ch=self.dense_in_ch,
                growth_rate=self.growth_rate,
                n_layers=self.n_layers
            )

            dense_out_ch = self.dense_in_ch + self.growth_rate * self.n_layers

            self.final_conv = nn.Sequential(
                nn.Conv2d(dense_out_ch, out_chans, 1),
                nn.BatchNorm2d(out_chans),
                nn.ReLU(inplace=True)
            )
            self.attention = SelfAttentionBlock(
                key_in_channels=out_chans,
                query_in_channels=out_chans,
                transform_channels=out_chans // 2,
                out_channels=out_chans,
                key_query_num_convs=2,
                value_out_num_convs=1
            )

        def forward(self, x, f4, f):  # inter[1], f3, f4
            # print(f"x channels: {x.shape}")
            y = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
            y = self.conv_small(y) #f3 512
            x = self.conv_big(x) # inter[1] 256
            # print(f"x channels: {x.shape}, y channels: {y.shape}") #512 40 20/1024 40 20
            f = self.conv1x1(f) #f4 1024变为256

            f = F.interpolate(f, scale_factor=4, mode='bilinear',
                                       align_corners=True)  #对 f4进行上采样，将其尺寸增加四倍，inter[1], f3, f4尺寸相同
            pj = torch.cat([x, y, f], dim=1)  # 512+1024  inter[0]和f4  512+256 inter[1]和f3 pj的通道数是512+256+256 40 20
            pj = self.conv_pj(pj)  # 256
            # print(f"pj channels: {pj.shape}") # 512
            new_x = self.attention(pj, pj, x)
            # print(f"new_x channels: {new_x.shape}") # new_x channels: 512
            # f3 = torch.cat([new_x, x], dim=1)  # 512 512 f3 1024 注释
            # f3 = self.up0(f3)  # 512 对拼接的特征图进行卷积得f3 变为512 注释
            # print(f"f3 channels: {f3.shape}")
            # 使用密集连接块
            dense_input = torch.cat([new_x, x], dim=1)
            dense_out = self.dense_block(dense_input)

            f3 = self.final_conv(dense_out)
            return f3 # f3 f2 f1 f0 分别是经过解码器加强后的inter0 1  3 6



# class Decoder_UP1(nn.Module):
#     def __init__(self, in_chans=512, out_chans=256, conv_op=nn.Conv2d):
#         super(Decoder_UP1, self).__init__()
#
#         self.up = double_conv_relu_batch(in_chans * 2, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
#         self.conv1x1 = single_conv_relu_batch(in_chans * 2, out_chans, kernel_size=1, stride=1, padding=0,
#                                               conv_op=conv_op)
#
#     def forward(self, x, f3, f4): # inter[1] f3 f4
#         f3 = F.interpolate(f3, scale_factor=2, mode='bilinear', align_corners=True)  #对f3进行上采样，将其尺寸增加一倍，使用双线性插值法，并确保角点对齐
#
#         f4 = self.conv1x1(f4) # f4变成256 f3 512 x 256
#
#         f4 = F.interpolate(f4, scale_factor=4, mode='bilinear',
#                            align_corners=True)  #对 f4进行上采样，将其尺寸增加四倍，使用双线性插值法，并确保角点对齐
#
#         x = torch.cat([x, f3, f4], dim=1)  # 128 x 3
#         f2 = self.up(x)  #将拼接后的特征 x传递给双卷积层进行特征提取和变换，得到最终的特征
#         return f2


# class Decoder_UP2(nn.Module):
#     def __init__(self, in_chans=256, out_chans=128, conv_op=nn.Conv2d):
#         super(Decoder_UP2, self).__init__()
#
#         self.conv1x1 = single_conv_relu_batch(in_chans * 2, out_chans, kernel_size=1, stride=1, padding=0,
#                                               conv_op=conv_op)
#         self.up = double_conv_relu_batch(out_chans * 5, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
#
#     def forward(self, x1, x2, f2, f3): # inter[2], inter[3], f2, f3
#         y = F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=True)
#
#         f3 = self.conv1x1(f3)
#         f3 = F.interpolate(f3, scale_factor=4, mode='bilinear', align_corners=True)
#         x = torch.cat([x1, x2, y, f3], dim=1)  # 64 64 32 16
#         f1 = self.up(x)
#         return f1


# class Decoder_UP3(nn.Module):
#     def __init__(self, in_chans=128, out_chans=64, conv_op=nn.Conv2d):
#         super(Decoder_UP3, self).__init__()
#
#         self.conv1x1 = single_conv_relu_batch(in_chans * 2, out_chans, kernel_size=1, stride=1, padding=0,
#                                               conv_op=conv_op)
#
#         self.up = double_conv_relu_batch(in_chans * 3, out_chans, kernel_size=3, padding=1, conv_op=conv_op)
#
#     def forward(self, x1, x2, x3, f1, f2):
#         f1 = F.interpolate(f1, scale_factor=2, mode='bilinear', align_corners=True)
#
#         f2 = self.conv1x1(f2)
#         f2 = F.interpolate(f2, scale_factor=4, mode='bilinear', align_corners=True)
#
#         x = torch.cat([x1, x2, x3, f1, f2], dim=1)
#         f0 = self.up(x)
#         return f0


class InstanceCounter(nn.Module):
    def __init__(self, input_n_filters, out_chs=16, usegpu=True):
        super(InstanceCounter, self).__init__()
        self.input_n_filters = input_n_filters  #将输入特征图的通道数保存为实例变量
        self.n_filters = out_chs
        self.out_filter = out_chs
        self.usegpu = usegpu
        self.output = nn.Sequential(nn.Linear(self.out_filter, 1),  # 该线性层接收 self.out_filter 维的输入，并输出 1 维的结果。
                                    nn.Sigmoid())
        self.cnn = nn.Sequential(
            nn.MaxPool2d(2, 2),
            single_conv_relu_batch(in_ch=self.input_n_filters, out_ch=self.out_filter, kernel_size=3, stride=1,
                                   padding=1),
            single_conv_relu_batch(in_ch=self.out_filter, out_ch=self.out_filter, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层处理后，x 的形状通常为 (batch_size, self.out_filter, 1, 1)
        )  #从输入特征图中提取特征

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(3).squeeze(2)  # nn.Linear 全连接层要求输入是二维张量  所以移除第3 4个维度为1的值
        x = self.output(x)
        return x


class DenseDecoderBlock(nn.Module):
    def __init__(self, in_ch, growth_rate=32, n_layers=4): # 每层新增的特征图数量 密集块中的层数
        super().__init__()
        self.layers = nn.ModuleList() # 用于存储各层卷积序列
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_ch + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch + i * growth_rate, growth_rate, 3, padding=1),
                nn.Dropout2d(0.1)  # 保持与您原有代码一致的dropout率
            ))

    def forward(self, x):
        features = [x] # 初始化特征列表
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


class RsUnet11(nn.Module):
    def __init__(self, in_chans=[64, 128, 256, 512, 1024], out_chs=20, out_class=2):
        super(RsUnet11, self).__init__()
        self.down = Resnet18(in_channels=3, out_channels=in_chans)  # 下采样
        self.ins_counter = InstanceCounter(in_chans[0], out_chs=out_chs)  # 计算实例数量
        self.ins_segHead = instanceSegmenationHead(in_chans[0], out_chans=out_chs)  # 用于进行实例分割
        self.sem_segHead = instanceSegmenationHead(in_chans[0], out_class)  # 用于进行语义分割

        self.decoder1 = Decoder_UP(in_chans[4], in_chans[3])  # 上采样操作
        # self.up1 = Decoder_UP1(in_chans[3], in_chans[2])
        # self.up2 = Decoder_UP2(in_chans[2], in_chans[1])
        # self.up3 = Decoder_UP3(in_chans[1], in_chans[0])
        self.decoder2 = Decoder_UP1(in_chans[3], in_chans[2]) #多拼接一个四采样的高层次特征图
        self.decoder3 = Decoder_UP1(in_chans[2], in_chans[1])
        self.decoder4 = Decoder_UP3(in_chans[1], in_chans[0]) #多拼接一个四采样的高层次特征图 更换注意力类型

        self.rc_skip = RC_skip(in_chans[2], in_chans[1])  #  256 128 用于特征融合


        self.apply(self._init_weight)  # 将 _init_weight 函数应用到模型的所有子模块上

        # self.side1 = nn.Conv2d(512, 64, kernel_size=1)
        # self.side2 = nn.Conv2d(256, 64, kernel_size=1)
        # self.side3 = nn.Conv2d(128, 64, kernel_size=1)
        # 添加side分支
        self.side1 = nn.Sequential(
            nn.Conv2d(512, 64, 1),  # 假设f3有512通道
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.side2 = nn.Sequential(
            nn.Conv2d(256, 64, 1),  # 假设f2有256通道
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.side3 = nn.Sequential(
            nn.Conv2d(128, 64, 1),  # 假设f1有128通道
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 添加可学习融合权重（关键修复）

        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)  # 4个输出融合

    def forward(self, x):

        features, x = self.down(x)  #接受输入x并返回下采样后的特征features和剩余的部分x  F3-512 F2-256 F1-128 F0-64,F4

        inter = []  # 存储下采样和上采样之间的中间特征
        inter.append(features[0])  #inter[0] F3

        s = self.rc_skip(features[3], features[2], features[1], features[0])  # F0 F1 F2 F3
        inter.extend(s)  # inter数组：F3  d3_1, d2_1, d2_2, d1_1, d1_2, d1_3
        # print(f"f4 channels: {x.shape}")
        # print(f"inter[0] channels: {inter[0].shape}")
        # print(f"inter[1] channels: {inter[1].shape}")
        # print(f"inter[2] channels: {inter[2].shape}")
        # print(f"inter[3] channels: {inter[3].shape}")
        # print(f"inter[4] channels: {inter[4].shape}")
        # print(f"inter[5] channels: {inter[5].shape}")
        # print(f"this is decoderthis is decoderthis is decoderthis is decoderthis is decoderthis is decoder")
        f4 = x  #将剩余的部分x赋值给f4 f4是F4
        f3 = self.decoder1(inter[0], x)  # 开始上采样
        out1 = F.interpolate(self.side1(f3), scale_factor=8, mode='bilinear') # 1修改最终输出的特征图---添加
        f2 = self.decoder2(inter[1], f3, f4)
        out2 = F.interpolate(self.side2(f2), scale_factor=4, mode='bilinear')# 2修改最终输出的特征图---添加
        f1 = self.decoder3(inter[3], f2, f3)
        out3 = F.interpolate(self.side3(f1), scale_factor=2, mode='bilinear')# 3修改最终输出的特征图---添加
        f0 = self.decoder4(inter[6], f1, f2)
        # 方案2：可学习权重融合（推荐） # 4修改最终输出的特征图---添加
        weights = torch.softmax(self.fusion_weights, dim=0)
        f0 = weights[0] * out3 + weights[1] * out2 + weights[2] * out1 + weights[3] * f0
        # f0=0.05 * out1 + 0.05 * out2 + 0.1 * out3+0.8* f0
        # f2 = self.up1(inter[1], f3, f4)
        # f1 = self.up2(inter[2], inter[3], f2, f3)
        # f0 = self.up3(inter[4], inter[5], inter[6], f1, f2)  # 后面模块对最终的上采样特征图f0 进行处理

        ins_pre = self.ins_segHead(f0)  # 得到实例分割的预测结果 ins_pre
        seg_pre = self.sem_segHead(f0)  # 得到语义分割的预测结果 seg_pre
        n_instance = self.ins_counter(f0)  #得到实例数量的预测结果 n_instance

        #return  seg_pre
        return (seg_pre, ins_pre, n_instance)
        #ins_pre, n_instance)



    def _init_weight(self, m):
        init_xavier = True
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  # 两种不同的权重初始化方式，
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
# Xavier 初始化方法的核心思想是保证每层的输入和输出的方差大致相同，这样能有效缓解梯度消失和梯度爆炸问题，让模型在训练过程中更稳定地收敛。
# normal_函数 He 初始化，它更适合于使用 ReLU 激活函数的网络，能让梯度在反向传播过程中更好地流动。
