from os import listdir
from os.path import join
import numpy
import torch
import Model as vo
import Grunet_koma_test as dic
from deepcoloring.data import train_Reader
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import GaussianBlur



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    basepath1 = "/Yx/majunze/fyj/GrscUnet/Dataset/minority_instance_dataset/train"
    train_rgb = sorted([join(basepath1, f) for f in listdir(basepath1) if f.endswith('_rgb.png')])
    train_labels = sorted([join(basepath1, f) for f in listdir(basepath1) if f.endswith('_label.png')])

    if 0 == len(train_rgb):
        print("No cvppp dataset found in:" + basepath1)
        exit(-1)

    numpy.random.seed(1203412413)
    indexes = numpy.random.permutation(len(train_rgb))   # 产生一个随机序列
    perm_rgb = numpy.array(train_rgb)[indexes].tolist()
    perm_labels = numpy.array(train_labels)[indexes].tolist()


    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         #transforms.RandomVerticalFlip(),
         transforms.RandomResizedCrop((320, 160), scale=(0.7, 1.0)),
         transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])


    transform_target = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
         transforms.RandomResizedCrop((320, 160), scale=(0.7, 1.0), interpolation=0),
         ])


    transform_test = transforms.Compose(
        [transforms.Resize((320, 160)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    transform_test_target=transforms.Compose([
        transforms.Resize((320, 160), interpolation=0)
    ])

    train_data = train_Reader(image_list=perm_rgb, labels_list=perm_labels, transform=transform, transform_target=transform_target,
                   use_cache=True)

    generator = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)

    # net = vo.RsUnet(out_chs=20).cuda()
    # net.load_state_dict(torch.load("mode.th"))
    # net.eval()
    
    net = vo.RsUnet11(out_chs = 12).cuda()
    net = dic.train(train_dataloder=generator, model=net, niter=200, max_n_objects=12)

    #net.load_state_dict(torch.load("RSUnet.th"))
    net.eval()

    basepath2 = "/Yx/majunze/fyj/GrscUnet/Dataset/minority_instance_dataset/test"
    test_rgbs = sorted([join(basepath2, f) for f in listdir(basepath2) if f.endswith('_rgb.png')])
    test_labels = sorted([join(basepath2, f) for f in listdir(basepath2) if f.endswith('_label.png')])

    test_data = train_Reader(image_list=test_rgbs, labels_list=test_labels, transform=transform_test,transform_target=transform_test_target, use_cache=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    dic.evalution(test_dataloader, net, max_n_objects=12, data_test_names=None)
