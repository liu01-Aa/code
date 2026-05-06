import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import torch
from itertools import chain
from PIL import Image, ImageFilter

def upsample_prediction(prediction, image_height, image_width):
    return cv2.resize(prediction, (image_width, image_height),
                      interpolation=cv2.INTER_NEAREST)

class Roation:
    def __init__(self,roation=90, resample=Image.NEAREST):
        self.roationx = roation
        if 1 == resample:
            self.resample = Image.BILINEAR
        else:
            self.resample = Image.NEAREST

    def __call__(self, x):
        # x = Image.open(x)
        x = x.rotate(self.roationx, resample=self.resample, expand=1)
        return x

class GaussianBlur:
    def __init__(self, sigma=[.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def best_dice(l_a, l_b):
    """
    Best Dice function
    :param l_a: list of binary instances masks
    :param l_b: list of binary instances masks
    :return: best dice estimation
    """
    result = 0
    for a in l_a:
        best_iter = 0
        for b in l_b:
            inter = 2 * float(np.sum(a * b)) / float(np.sum(a) + np.sum(b))
            if inter > best_iter:
                best_iter = inter
        result += best_iter
    if 0 == len(l_a):
        return 0
    return result / len(l_a)

def symmetric_best_dice(l_ar, l_gr):
    """
    Symmetric Best Dice function
    :param l_ar: list of output binary instances masks
    :param l_gr: list of binary ground truth masks
    :return: Symmetric best dice estimation
    """
    return np.min([best_dice(l_ar, l_gr), best_dice(l_gr, l_ar)])


def calc_dic(n_objects_gt, n_objects_pred):
    return  n_objects_gt - n_objects_pred

def calc_dice(gt_seg, pred_seg):


    gt_seg = (gt_seg == 1).astype('bool')
    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice

def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    print(gt_object_idxes)
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))
    print(pred_object_idxes)

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        print(dices)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice

def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)


def process_sem_nbj(sem_pred_, n_objects_, max_n_objects):
    sem_pred_ = torch.nn.functional.softmax(sem_pred_, dim=1).detach().cpu().numpy()

    n_objects_ = n_objects_ * max_n_objects
    n_objects_ = torch.round(n_objects_).int()
    n_objects_ = n_objects_.squeeze(0).detach().cpu().numpy()
    if len(n_objects_) != 1:
        n_objects_ = list(chain.from_iterable(n_objects_))

    return sem_pred_, n_objects_


def gen_mask(ins_img):
    mask = []
    for i, mask_i in enumerate(ins_img):
        binarized = mask_i * (i + 1)
        mask.append(binarized)
    mask = np.sum(np.stack(mask, axis=0), axis=0).astype(np.uint8)
    return mask


def coloring(mask):    #给实例分割掩码上色
    ins_color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    n_ins = len(np.unique(mask)) - 1
    # n_ins = len(np.unique(mask))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_ins)]
    for i in range(n_ins):
        ins_color_img[mask == i + 1] =\
            (np.array(colors[i][:3]) * 255).astype(np.uint8)
    return ins_color_img, mask


def gen_instance_mask(sem_pred, ins_pred, n_obj):

    embeddings = ins_pred[:, sem_pred].transpose(1, 0)

    clustering = KMeans(n_clusters=n_obj, n_init=20, max_iter=500).fit(embeddings)
    labels = clustering.labels_


    instance_mask = np.zeros_like(sem_pred, dtype=np.uint8)
    for i in range(n_obj):
        lbl = np.zeros_like(labels, dtype=np.uint8)
        lbl[labels == i] = i + 1
        instance_mask[sem_pred] += lbl


    return instance_mask


def gen_color_img(sem_pred, ins_pred, n_obj):
    return coloring(gen_instance_mask(sem_pred, ins_pred, n_obj))

def visualization(sou_images, se_masks, ins_masks, cout = 0):
    plt.gray()
    for i in range(len(ins_masks)):
        plt.imshow(sou_images[i])
        plt.savefig("dress0" + str(cout) + str(i) + "_rgb_pre.png")

        plt.imshow(ins_masks[i])
        plt.savefig("dress0" + str(cout) + str(i) + "_ins_pre.png")

        plt.imshow(se_masks[i])
        plt.savefig("dress0" + str(cout) + str(i) + "_fg_pre.png")

def _get_ins_seg_masks(sem_pred, ins_pred, n_objects):


    sem_pred = np.concatenate(sem_pred)[:, 1, :, :]
    #ins_pred = np.concatenate(ins_pred)   #对实例分割预测进行连接操作


    # Post Processing
    p_sem_pred = []
    op = 0.5
    for sp in sem_pred:

        sem_mask = np.zeros((sp.shape[0], sp.shape[1]), dtype='bool')
        sem_mask[sp > op] = True
        sem_mask[sp < op] = False
        sem_mask[sp == op] = True
        p_sem_pred.append(sem_mask)    #存储生成的二值掩码

    ins_masks = []
    ins_color_imgs = []



    for i in range(len(p_sem_pred)):

        color_img, ins_mask = gen_color_img(p_sem_pred[i], ins_pred[i], n_objects[i])
        ins_masks.append(ins_mask)
        ins_color_imgs.append(color_img)

    return  p_sem_pred, ins_masks, ins_color_imgs




def evaluate(gt_n_objects=None,pre_n_objects=None):





    effdics = calc_dic(gt_n_objects[0], pre_n_objects[0])




    return effdics

def eval(gt_ins_mask, gt_label):
    sbd = calc_sbd(gt_ins_mask, gt_label.numpy())
    return sbd


def IOUMetric(label_pred, label_true, num_class):
    mask = (label_true >= 0) & (label_true < num_class)
    hist = np.bincount(num_class*label_true[mask].astype('int') + label_pred[mask],
                       minlength=num_class ** 2).reshape(num_class, num_class)

    Iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))

    return Iou


##############################################################################
def DiffFGLabels(inLabel, gtLabel):

    if (inLabel.shape != gtLabel.shape):
        return -1

    maxInLabel = np.int(np.max(inLabel))  # maximum label value in inLabel
    minInLabel = np.int(np.min(inLabel))  # minimum label value in inLabel
    maxGtLabel = np.int(np.max(gtLabel))  # maximum label value in gtLabel
    minGtLabel = np.int(np.min(gtLabel))  # minimum label value in gtLabel

    return (maxInLabel - minInLabel) - (maxGtLabel - minGtLabel)


##############################################################################
def BestDice(inLabel, gtLabel):


    score = 0  # initialize output

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return score

    maxInLabel = np.max(inLabel)  # maximum label value in inLabel
    minInLabel = np.min(inLabel)  # minimum label value in inLabel
    maxGtLabel = np.max(gtLabel)  # maximum label value in gtLabel
    minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel

    if (maxInLabel == minInLabel):  # trivial solution
        return score

    for i in range(minInLabel + 1, maxInLabel + 1):  # loop all labels of inLabel, but background
        sMax = 0;  # maximum Dice value found for label i so far
        for j in range(minGtLabel + 1, maxGtLabel + 1):  # loop all labels of gtLabel, but background
            s = Dice(inLabel, gtLabel, i, j)  # compare labelled regions
            # keep max Dice value for label i
            if (sMax < s):
                sMax = s
        score = score + sMax;  # sum up best found values
    score = score / (maxInLabel - minInLabel)
    return score


##############################################################################
def FGBGDice(inLabel, gtLabel):
    # input: inLabel: label image to be evaluated. Background label is assumed to be the lowest one.
    #        gtLabel: ground truth label image. Background label is assumed to be the lowest one.
    # output: Dice score for foreground/background segmentation, only.

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return 0

    minInLabel = np.min(inLabel)  # minimum label value in inLabel
    minGtLabel = np.min(gtLabel)  # minimum label value in gtLabel

    one = np.ones(inLabel.shape)
    inFgLabel = (inLabel != minInLabel * one) * one
    gtFgLabel = (gtLabel != minGtLabel * one) * one

    return Dice(inFgLabel, gtFgLabel, 1, 1)  # Dice score for the foreground


##############################################################################
def Dice(inLabel, gtLabel, i, j):
    # calculate Dice score for the given labels i and j

    # check if label images have same size
    if (inLabel.shape != gtLabel.shape):
        return 0

    one = np.ones(inLabel.shape)
    inMask = (inLabel == i * one)  # find region of label i in inLabel
    gtMask = (gtLabel == j * one)  # find region of label j in gtLabel
    inSize = np.sum(inMask * one)  # cardinality of set i in inLabel
    gtSize = np.sum(gtMask * one)  # cardinality of set j in gtLabel
    overlap = np.sum(inMask * gtMask * one)  # cardinality of overlap of the two regions
    if ((inSize + gtSize) > 1e-8):
        out = 2 * overlap / (inSize + gtSize)  # Dice score
    else:
        out = 0

    return out


##############################################################################
def AbsDiffFGLabels(inLabel, gtLabel):
    # input: inLabel: label image to be evaluated. Labels are assumed to be consecutive numbers.
    #        gtLabel: ground truth label image. Labels are assumed to be consecutive numbers.
    # output: Absolute value of difference of the number of foreground labels

    return np.abs(DiffFGLabels(inLabel, gtLabel))



def LSC_evalute(inLabel, gtLabel):

    # diff = DiffFGLabels(inLabel, gtLabel)
    bd = BestDice(inLabel, gtLabel)

    sbd = min(BestDice(inLabel, gtLabel), BestDice(gtLabel, inLabel))
    fg_dice = FGBGDice(inLabel, gtLabel)
    return  fg_dice, bd, sbd
