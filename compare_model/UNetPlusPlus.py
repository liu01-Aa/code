import torch
import torch.nn as nn
class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm = True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
 
        if pre_Batch_Norm:
          self.Conv_forward = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
 
        else:
          self.Conv_forward = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU())
 
    def forward(self, x):
        x = self.Conv_forward(x)
        return x

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
        self.ins_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, 0))

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
    def __init__(self, in_chs=64, out_chs=32):
        super(feHead, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_chs),
                                  nn.ReLU(inplace=True))
        self.ins_head = instanceSegmenationHead(in_chans=out_chs, out_chans=10)
        self.seg_head = instanceSegmenationHead(in_chans=out_chs, out_chans=2)
        self.counter = InstanceCounter(input_n_filters=out_chs, out=10)
    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        ins_head = self.ins_head(x)
        seg_head = self.seg_head(x)
        count = self.counter(x)
        return seg_head, ins_head, count



class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]
        
        self.CONV3_1 = ContinusParalleConv(512*2, 512, pre_Batch_Norm = True)
 
        self.CONV2_2 = ContinusParalleConv(256*3, 256, pre_Batch_Norm = True)
        self.CONV2_1 = ContinusParalleConv(256*2, 256, pre_Batch_Norm = True)
 
        self.CONV1_1 = ContinusParalleConv(128*2, 128, pre_Batch_Norm = True)
        self.CONV1_2 = ContinusParalleConv(128*3, 128, pre_Batch_Norm = True)
        self.CONV1_3 = ContinusParalleConv(128*4, 128, pre_Batch_Norm = True)
 
        self.CONV0_1 = ContinusParalleConv(64*2, 64, pre_Batch_Norm = True)
        self.CONV0_2 = ContinusParalleConv(64*3, 64, pre_Batch_Norm = True)
        self.CONV0_3 = ContinusParalleConv(64*4, 64, pre_Batch_Norm = True)
        self.CONV0_4 = ContinusParalleConv(64*5, 64, pre_Batch_Norm = True)
 
 
        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm = False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm = False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm = False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm = False)
        
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm = False)
 
        self.pool = nn.MaxPool2d(2)
    
        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) 
 
        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) 
 
        
        # 分割头
        self.final_super_0_1 = feHead(in_chs=64, out_chs=num_classes)

        self.final_super_0_2 = feHead(in_chs=64, out_chs=num_classes)
        self.final_super_0_3 = feHead(in_chs=64, out_chs=num_classes)
        self.final_super_0_4 = feHead(in_chs=64, out_chs=num_classes)
 
        
    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))
        
        x_0_1 = torch.cat([self.upsample_0_1(x_1_0) , x_0_0], 1)
        x_0_1 =  self.CONV0_1(x_0_1)
        
        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)
        
        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)
        
        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)
 
        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)
        
        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)
        
        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)
 
        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)
        
        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)
        
        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)
    
    
        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            seg_head, ins_head, count = self.final_super_0_4(x_0_4)
            return seg_head, ins_head, count
 
 
if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    model = UnetPlusPlus(num_classes=3, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    print(outputs.shape)    
    
    print("deep_supervision: True")
    deep_supervision = True
    model = UnetPlusPlus(num_classes=3, deep_supervision=deep_supervision).to(device)
    outputs = model(inputs)
    for out in outputs:
        print(out.shape)
 
 