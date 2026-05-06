import sys
sys.path.insert(0, '../..')
import torch
import deepcoloring as dc
import numpy as np
from PIL import  Image
from .utils import _get_ins_seg_masks, evaluate, LSC_evalute, symmetric_best_dice, process_sem_nbj, visualization, eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evalution(test_dataloader, model, max_n_objects=20, data_test_names=None):
    model.eval()
    print("Model loaded")
    mask_builder = dc.Discriminative_binary_Mask(max_n_objects=max_n_objects, sem_class=2)
    sem_pred = []
    ins_pred = []

    sem_pred_gt = []
    ins_pred_gt = []
    cout = 1
    dics, sbds, fg_dices = [], [], []
    bds = []

    dir_in = "/Yx/majunze/fyj/GrscUnet/Grunet_koma_test/ReUNet1"
    for batch in test_dataloader:
        images_, gt_labels_ = batch

        images_ = images_.cuda()
        labels_ = gt_labels_
        sem_pred_, ins_pred_, n_objects_ = model(images_)
        instance_mask, sem_mask, gt_objects = mask_builder(labels_)

        instance_mask = instance_mask.cpu().numpy()
        sem_mask = sem_mask.cpu().numpy()
        gt_objects_ = gt_objects.cpu().numpy()

        sem_pred_gt.append(sem_mask)
        ins_pred_gt.append(instance_mask)
        p_sem_pred_gt, ins_masks_gt, ins_color_imgs_gt = _get_ins_seg_masks(sem_pred_gt, ins_pred_gt, gt_objects_)

        ins_pred_ = ins_pred_.detach().cpu().numpy()

        sem_pred_, n_objects_ = process_sem_nbj(sem_pred_, n_objects_, max_n_objects)
        gt_objects_ = gt_objects.cpu().numpy()
        sem_pred.append(sem_pred_)
        ins_pred.append(ins_pred_)

        print("pre_N: {}, gt_N:{}".format(n_objects_[0], gt_objects_[0]))

        p_sem_pred, ins_masks, ins_color_imgs = _get_ins_seg_masks(sem_pred, ins_pred, n_objects_)
      

        diff =  evaluate( gt_n_objects=gt_objects_, pre_n_objects=n_objects_)
        fg_dice, bd, sbd = LSC_evalute(ins_masks[0], ins_masks_gt[0])

        print("effdic={}, fg_dices={:.3f}%, bd={:.3f}%, sbd={:.3f}%".format(diff, fg_dice*100, bd*100, sbd*100))


        dics.append(diff)
        fg_dices.append(fg_dice)
        bds.append(bd)
        sbds.append(sbd)

        sem_pred = []
        ins_pred = []

        sem_pred_gt = []
        ins_pred_gt = []


        ins_masks = Image.fromarray(np.uint8(ins_color_imgs[0]))
        p_sem_pred = np.asarray(p_sem_pred[0]) * 255
        p_sem_pred = Image.fromarray(np.uint8(p_sem_pred))

        ins_masks.save('dress_RSUnet_' + str(cout) + '_ins_label.png')
        p_sem_pred.save('dress_RSUnet_' + str(cout) + '_seg_label.png')
        cout = cout + 1

    print("hhhhhhhh")
    print("diffFG={}".format(np.mean(dics)))
    print("absDiffFG={}".format(np.mean(np.abs(dics))))
    print("FgBgDice={:.3f}%".format(np.mean(fg_dices)*100))
    print("bestDice={:.3f}%".format(np.mean(bds)*100))
    print("SBD={:.3f}%".format(np.mean(sbds)*100))

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # print(label_true.is_cuda)
        # print(self.num_classes.is_cuda)

        mask = (label_true >= 0) & (label_true < self.num_classes)
        # print(mask.shape)


        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        # print(len(predictions))
        print(predictions.shape)
        print(gts.shape)
        exit()
        for i in range(len(predictions)):
            self.hist += self._fast_hist(predictions[i].flatten(), gts[i].flatten())
        # for lp, lt in zip(predictions, gts):
        #     i = i + 1
        #     self.hist += self._fast_hist(lp.flatten(), lt.flatten())


    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def label_mask(label):
    label_height, label_width = label.shape

    instance_values = set(np.unique(label)).difference([0])
    n_instances = len(instance_values)
    # print(n_instances)
    print(instance_values)
    instance_mask = np.zeros((n_instances, label_height, label_width), dtype=np.uint8)
    for l, v in enumerate(instance_values):
        _mask = np.zeros((label_height, label_width), dtype=np.uint8)
        _mask[label == v] = 1
        instance_mask[l, :, :] = _mask


def hist(label_pred, label_true, num_classes):
    # print(label_true.is_cuda)
    # print(self.num_classes.is_cuda)

    mask = (label_true >= 0) & (label_true < num_classes)
    # print(mask.shape)
    print(np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2).shape)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2).reshape(
        num_classes, num_classes)
    return hist

 # source_img = Image.fromarray(np.uint8(source_img[0]))
        # ins_masks = Image.fromarray(np.uint8(ins_color_imgs[0]))
        #
        # p_sem_pred = np.asarray(p_sem_pred[0]) * 255
        # p_sem_pred = Image.fromarray(np.uint8(p_sem_pred))
        #
        # source_img.save(dir_in + 'plant_' + str(cout) + '_rgb.png')
        # ins_masks.save(dir_in + 'plant_' + str(cout) + '_ins_label.png')
        # p_sem_pred.save(dir_in + 'plant_' + str(cout) + '_seg_label.png')