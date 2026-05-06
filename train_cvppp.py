from os import listdir
from os.path import join
import numpy
import torch
import compare_model as db
import Model as vo
import Grunet_koma_test as dic
from deepcoloring.data import train_Reader
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import GaussianBlur



# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    basepath =  r"/tmp/pycharm_project_425/code4.3/code/Dataset/CVPPP/A1"
    rgb = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])

    if 0 == len(rgb):
        print("No cvppp dataset found in:" + basepath)
        exit(-1)

    numpy.random.seed(1203412413) # 对于实验的可重复性非常重要
    indexes = numpy.random.permutation(len(rgb))   # 产生一个随机序列 0 到 len(rgb) - 1的整数
    perm_rgb = numpy.array(rgb)[indexes].tolist()  # rgb初始顺序列表先转换为numpy数组，再利用随机顺序的索引把图片打乱顺序 再放进列表
    perm_labels = numpy.array(labels)[indexes].tolist()


    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), # 水平翻转
         transforms.RandomVerticalFlip(), # 垂直翻转
         transforms.RandomResizedCrop(256, scale=(0.7, 1.0)), # 3. 随机大小裁剪
         transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5), # 4. 随机应用高斯模糊
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])


    transform_target = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomResizedCrop(256, scale=(0.7, 1.0), interpolation=0),
         ])

# 以上操作对图像和标签进行了数据增强  增加了训练数据的多样性，有助于模型学习到更鲁棒的特征，从而提高模型的泛化能力。
    transform_test = transforms.Compose(
        [transforms.Resize((256, 256)),   # 统一输入图像的尺寸
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    transform_test_target=transforms.Compose([
        transforms.Resize((256, 256), interpolation=0)
    ])

    train_data = train_Reader(image_list=perm_rgb[:-20], labels_list=perm_labels[:-20], transform=transform, transform_target=transform_target,
                   use_cache=True)  #  训练数据是除了后20个图片的所有图片

 # batch_siz 8修改为4
    generator = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)

    # net = vo.RsUnet(out_chs=20).cuda()
    # net.load_state_dict(torch.load("mode.th"))
    # net.eval()
    # net = db.UNet(out_chs=20).to(device)
    net = vo.RsUnet11(out_chs=20).to(device)
    net = dic.train(train_dataloder=generator, model=net, niter=5, max_n_objects=20)
    # 用train 方法来对模型进行训练 模型从 generator 中获取批量的训练数据 对net这个模型（也就是前面指定的RsUnet11）进行训练
    # net.load_state_dict(torch.load("RSUnet01.th"))
    # net.eval() # 通常会在需要对模型进行评估（验证、测试、推理）之前被调用
    # 在完成评估后，如果需要继续训练模型，还需要调用 net.train() 将模型切换回训练模式。

    # basepath = "../Dataset/komaa/test"
    # t_rgbs = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])
    # t_labels = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_label.png')])

    test_data = train_Reader(image_list=perm_rgb[-20:], labels_list=perm_labels[-20:], transform=transform_test,transform_target=transform_test_target, use_cache=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

    dic.evalution(test_dataloader, net, max_n_objects=20, data_test_names=None)  # evalution：dic 对象的一个方法，用于对模型进行评估。

# hhhhhhhh
# diffFG=-0.1875
# absDiffFG=0.1875
# FgBgDice=98.951%
# bestDice=96.508%
# SBD=95.789%
