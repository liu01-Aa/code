import os
import os.path
import random
import numpy as np
import torch
from torchvision import transforms

# from utils import ImageUtilities as IU
import torch.utils.data as data
from PIL import Image
import torch.nn as nn

from  skimage.io import imread
import skimage.transform as transform


def default_loader(filepath):                       #加载图像文件
    return Image.open(filepath).convert('RGB')      #将图像转换为RGB模式
    # return Image.open(filepath)

def default_label(filepath):                        #加载标签图像文件
    return Image.open(filepath).convert('L')        #将标签图像转换为灰度模式

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SSSReader(data.Dataset):
    def __init__(self, image_list, labels_list, transform=None, use_cache=True,
                 loader=default_loader, scale=2):

        self.images = image_list
        self.loader = loader

        if len(labels_list) != 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list

        else:
            self.labels = False

        self.transform = transform
        self.cache = {}                      #创建一个空字典，用于存储图像数据的缓存。
        self.use_cache = use_cache
        self.scale = scale                   #scale是一个整数，表示缩放因子，默认为2，用于缩小图像和标签

    def __len__(self):
        return len(self.images)              #返回数据集中样本的数量

    def __getitem__(self, idx):
        if idx not in self.cache:
            rgb = imread(self.images[idx], plugin='pil')      #获取数据集中特定索引 idx处的样本
            # print(rgb.shape)  # (530,500,3)
            rgb = transform.resize(rgb, np.array(rgb.shape[:2]) / self.scale, mode='constant')
            # print(rgb.shape) #(265,250,3)
            label = None
            if idx < len(self.labels):
                label = imread(self.labels[idx], True, plugin='pil')
                label = np.digitize(label, bins=np.unique(
                    label)) - 1  # digitize()返回一个索引的数组,属于数组的每个值的bin的索引数组 unique()该函数是去除数组中的重复数字，并进行排序之后输出
                # print(label.shape) #(530,500) H,W
                label = label[::self.scale, ::self.scale]
                # print(label.shape) # (265, 250)
                label = label[:rgb.shape[0], :rgb.shape[1]]
                # print(label.shape) (265,250)
            # self.cache[idx] = (rgb.astype(np.float32), label.astype(np.int32))
        else:
            rgb, label = self.cache[idx]
        if self.use_cache:
            self.cache[idx] = (rgb, label)



        for t in self.transform:
            # random.randint用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <=
            seed = np.random.randint(2147483647)
            random.seed(seed)
            rgb = t(rgb, True)
            np.random.seed(seed)
            label = t(label)
        # if self.transform is not None:
        #     img = self.transform(rgb)
        # random.seed(seed)
        # if self.labels:
        #     if self.transform is not None:
        #         target = self.transform(label)
        return np.array(rgb), np.array(label)




class ssReader(data.Dataset):
    def __init__(self, model='training', image_list=[], labels_list=[],fgs_list =[], transform=None, transform_target=None, use_cache=True,
                 loader=default_loader):

        self.images = image_list
        self.loader = loader
        self.model = model

        if len(labels_list) != 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list

        else:
            self.labels = False

        if len(fgs_list) != 0:
            assert len(image_list) == len(fgs_list)
            self.fgs = fgs_list

        else:
            self.fgs = False

        self.transform = transform
        self.transform_target = transform_target
        # self.transform_Normalizer = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.521, 0.389, 0.206], std=[0.212, 0.151, 0.113])
        # ])
        self.trans = transforms.Compose(
            [transforms.Resize([530, 500])])
        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx])
            # img = np.asarray(img)
            # img = img.transpose(1, 0, 2)
            # img = Image.fromarray(img)


            if self.labels:
                target = Image.open(self.labels[idx])
                # img = np.asarray(img)
                # img = img.transpose(1, 0, 2)
                # img = Image.fromarray(img)
                # print(target.size)
            else:
                target = None

            if self.fgs:
                fg = Image.open(self.fgs[idx])
            else:
                fg = None
        else:
            img, target, fg= self.cache[idx]

        if self.use_cache:
            self.cache[idx] = (img, target, fg)

        if self.model == 'training':
            seed = np.random.randint(2147483647)
            # random.seed(seed)
            if self.transform is not None:
                if self.transform_target is not None:
                    setup_seed(seed)
                    img = self.transform(img)
                    setup_seed(seed)
                    target = self.transform_target(target)
            return np.array(img), np.array(target)
        elif self.model=='testing':
            source = img
            seed = np.random.randint(2147483647)

            # random.seed(seed)
            if self.transform is not None:
                if self.transform_target is not None:
                    setup_seed(seed)
                    img = self.transform(img)

                    # setup_seed(seed)
                    # source = self.transform_target(source)
                    # setup_seed(seed)
                    # target1 = self.transform_target(target)
                    # setup_seed(seed)
                    # fg = self.trans(target)

            # return  np.array(img)
            return np.array(source), np.array(img), np.array(target), np.array(fg)
        else:
            seed = np.random.randint(2147483647)
            # random.seed(seed)
            if self.transform is not None:
                    setup_seed(seed)
                    img = self.transform(img)
            # return  np.array(img)
            return  np.array(img)




class Reader_DATA(data.Dataset):
    def __init__(self, model='training', image_list=[], labels_list=[],fgs_list =[], transform=None, transform_target=None, use_cache=True,
                 loader=default_loader):

        self.images = image_list
        self.loader = loader
        self.model = model

        if len(labels_list) != 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list

        else:
            self.labels = False

        if len(fgs_list) != 0:
            assert len(image_list) == len(fgs_list)
            self.fgs = fgs_list

        else:
            self.fgs = False

        self.transform = transform
        self.transform_target = transform_target
        # self.transform_Normalizer = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.521, 0.389, 0.206], std=[0.212, 0.151, 0.113])
        # ])

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx])
            # print(img.size)
            if self.labels:
                target = Image.open(self.labels[idx])
                # print(target.size)
            else:
                target = None

            if self.fgs:
                fg = Image.open(self.fgs[idx])
            else:
                fg = None
        else:
            img, target, fg= self.cache[idx]

        if self.use_cache:
            self.cache[idx] = (img, target, fg)

        if self.model == 'training':
            seed = np.random.randint(2147483647)
            # random.seed(seed)
            if self.transform is not None:
                if self.transform_target is not None:
                    setup_seed(seed)
                    img = self.transform(img)
                    setup_seed(seed)
                    target = self.transform_target(target)
            return np.array(img), np.array(target)
        elif self.model=='testing':
            source = img
            seed = np.random.randint(2147483647)

            # random.seed(seed)
            if self.transform is not None:
                if self.transform_target is not None:
                    setup_seed(seed)
                    img = self.transform(img)

                    setup_seed(seed)
                    source = self.transform_target(source)
                    setup_seed(seed)
                    target1 = self.transform_target(target)
                    setup_seed(seed)
                    fg = target

            # return  np.array(img)
            return np.array(source), np.array(img), np.array(target1), np.array(fg)
        else:
            seed = np.random.randint(2147483647)
            # random.seed(seed)
            if self.transform is not None:
                if self.transform_target is not None:
                    setup_seed(seed)
                    img = self.transform(img)
            # return  np.array(img)
            return  np.array(img)


class train_Reader(data.Dataset):
    def __init__(self, image_list=[], labels_list=[], transform=None,
                 transform_target=None, use_cache=True,
                 loader=default_loader):

        self.images = image_list
        self.loader = loader


        if len(labels_list) != 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list

        else:
            self.labels = False

        self.transform = transform
        self.transform_target = transform_target

        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx])
            if self.labels:
                target = Image.open(self.labels[idx]) #原本

                # target = default_label(self.labels[idx])

            else:
                target = None
        else:
            img, target= self.cache[idx]

        if self.use_cache:
            self.cache[idx] = (img, target)

        seed = np.random.randint(2147483647)
            # random.seed(seed)
        if self.transform is not None:
            if self.transform_target is not None:
                setup_seed(seed)
                img = self.transform(img)
                setup_seed(seed)
                target = self.transform_target(target)

        return np.array(img), np.array(target)

class valid_Reader(data.Dataset):
    def __init__(self, image_list=[], labels_list=[], transform=None, use_cache=True,
                 loader=default_loader):

        self.images = image_list
        self.loader = loader


        if len(labels_list) != 0:
            assert len(image_list) == len(labels_list)
            self.labels = labels_list

        else:
            self.labels = False

        self.transform = transform
        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx]) # (500, 530) (w, h)
            # img = Image.open(self.labels[idx])
            source = img
            if self.labels:
                target = Image.open(self.labels[idx])
                # target = default_label(self.labels[idx])
            else:
                target = None
        else:
            source, img, target= self.cache[idx]
        if self.use_cache:
            self.cache[idx] = (source, img, target)
        seed = np.random.randint(2147483647)
            # random.seed(seed)
        if self.transform is not None:
            setup_seed(seed)
            img = self.transform(img)

        return np.array(source), np.array(img), np.array(target)





class test_DATA(data.Dataset):
    def __init__(self, image_list=[],  transform=None,
                  use_cache=True,loader=default_loader):

        self.images = image_list
        self.loader = loader
        self.transform = transform
        self.cache = {}
        self.use_cache = use_cache

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if idx not in self.cache:
            img = self.loader(self.images[idx]) #(530, 500) (w, h)
            source = img #(500, 530) (w, h)
        else:
            source, img = self.cache[idx]

        if self.use_cache:
            self.cache[idx] = (source, img)
        seed = np.random.randint(2147483647)
        if self.transform is not None:
                setup_seed(seed)
                img = self.transform(img)
        return np.array(source), np.array(img)

import random
from scipy import ndimage
def Random_rotate(image, label):
    angle = [-270, -240, -210, -180, -150, -120, -90, -60, -30, 30, 60, 90, 120, 150, 180, 210, 240, 270]
    i = random.randint(0, 17)
    image = np.asarray(image)
    label = np.asarray(label)

    image = ndimage.rotate(image, angle[i], order=0, reshape=False)
    label = ndimage.rotate(label, angle[i], order=0, reshape=False)

    image = Image.fromarray(np.uint8(image))
    label = Image.fromarray(np.uint8(label))

    return image, label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label