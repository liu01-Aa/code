import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.modules.loss import  _Loss
from torch.autograd import Variable
from torch import cosh
def flatten(x):
    return x.view(x.size(0), -1)


device = torch.device("cuda:0")
class Discriminative_Mask(nn.Module):
    def __init__(self, max_n_objects=25, sem_class=2):
        super(Discriminative_Mask, self).__init__()
        self.max_n_objects = max_n_objects
        self.sem_class = sem_class

    def _single_discriminative_mask(self, label):
        label_height, label_width = label.shape
        instance_values = set(np.unique(label)).difference([0])
        n_instances = len(instance_values)
        # print(n_instances)

        instance_mask = np.zeros((n_instances, label_height, label_width), dtype=np.uint8)
        for l, v in enumerate(instance_values):
            _mask = np.zeros((label_height, label_width), dtype=np.uint8)
            _mask[label == v] = 1
            instance_mask[l, :, :] = _mask

        instance_annotation_resized = []
        for i in range(n_instances):
            instance_ann_img = Image.fromarray(instance_mask[i, :, :])
            instance_ann_img = np.array(instance_ann_img)
            instance_annotation_resized.append(instance_ann_img)

        # 使用0填充实例注解
        for i in range(self.max_n_objects - n_instances):
            zero = np.zeros((label_height, label_width), dtype=np.uint8)
            zero = Image.fromarray(zero)
            zero = np.array(zero)
            instance_annotation_resized.append(zero.copy())

        sematic_mask = instance_mask.sum(0)
        sematic_mask[sematic_mask != 0] = 1
        sematic_mask = sematic_mask.astype(np.uint8)

        instance_mask = np.stack(instance_annotation_resized, axis=0)

        return instance_mask, sematic_mask, n_instances

    def forward(self, labels):
        batch, label_height, label_width = labels.shape

        instance_mask = [0] * batch
        sem_mask = [0] * batch
        n_object_list = [0] * batch

        for i in range(batch):
            _instance_mask, _sem_mask, _n_instance = self._single_discriminative_mask(labels[i])
            instance_mask[i] = _instance_mask
            sem_mask[i] = _sem_mask
            n_object_list[i] = _n_instance

        instance_mask = np.array(instance_mask, dtype='int')
        instance_mask = torch.LongTensor(instance_mask).to(device)

        sematic_annotatios = np.array(sem_mask, dtype='int')
        sematic_annotatios_one_hot = np.eye(self.sem_class, dtype='int')
        sematic_annotatios_one_hot = sematic_annotatios_one_hot[sematic_annotatios.flatten()].reshape(sematic_annotatios.shape[0],sematic_annotatios.shape[1],sematic_annotatios.shape[2], self.sem_class)
        sematic_annotatios_one_hot = torch.LongTensor(sematic_annotatios_one_hot)
        sematic_annotatios_one_hot = sematic_annotatios_one_hot.permute(0, 3, 1, 2)
        sematic_annotatios_one_hot = sematic_annotatios_one_hot.to(device)
        n_object_list = torch.LongTensor(n_object_list).to(device)
        return instance_mask, sematic_annotatios_one_hot, n_object_list

class Discriminative_binary_Mask(nn.Module):
    def __init__(self, max_n_objects=25, sem_class=2):
        super(Discriminative_binary_Mask, self).__init__()
        self.max_n_objects = max_n_objects
        self.sem_class = sem_class

    def _single_discriminative_mask(self, label):
        label_height, label_width = label.shape
        instance_values = set(np.unique(label)).difference([0])
        n_instances = len(instance_values)
        # print(n_instances)

        instance_mask = np.zeros((n_instances, label_height, label_width), dtype=np.uint8)
        for l, v in enumerate(instance_values):
            _mask = np.zeros((label_height, label_width), dtype=np.uint8)
            _mask[label == v] = 1
            instance_mask[l, :, :] = _mask

        instance_annotation_resized = []
        for i in range(n_instances):
            instance_ann_img = Image.fromarray(instance_mask[i, :, :])
            instance_ann_img = np.array(instance_ann_img)
            instance_annotation_resized.append(instance_ann_img)

        # 使用0填充实例注解
        for i in range(self.max_n_objects - n_instances):
            zero = np.zeros((label_height, label_width), dtype=np.uint8)
            zero = Image.fromarray(zero)
            zero = np.array(zero)
            instance_annotation_resized.append(zero.copy())


        sem_mask = np.zeros((label_height, label_width), dtype=bool)
        sem_mask[np.sum(instance_mask, axis=0) != 0] = True
        sem_mask = np.stack([~sem_mask, sem_mask]).astype(np.uint8)

        instance_mask = np.stack(instance_annotation_resized, axis=0)

        return instance_mask, sem_mask, n_instances

    def forward(self, labels):
        batch, label_height, label_width = labels.shape

        instance_mask = [0] * batch
        sem_mask = [0] * batch
        n_object_list = [0] * batch

        for i in range(batch):
            _instance_mask, _sem_mask, _n_instance = self._single_discriminative_mask(labels[i])
            instance_mask[i] = _instance_mask
            sem_mask[i] = _sem_mask
            n_object_list[i] = _n_instance

        instance_mask = np.array(instance_mask, dtype='int')
        instance_mask = torch.LongTensor(instance_mask).to(device)
        sem_mask = np.array(sem_mask, dtype='bool')  # res_model.th
        sem_mask = torch.LongTensor(sem_mask).to(device)
        n_object_list = torch.LongTensor(n_object_list).to(device)
        return instance_mask, sem_mask, n_object_list


class DiscriminativeLoss(_Loss):

    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True, size_average=True):
        super(DiscriminativeLoss, self).__init__(size_average)
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        #_assert_no_grad(target)
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        max_n_clusters = target.size(1)

        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg
        loss = loss

        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, : n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, : n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / target_sample.sum(2)

            # padding
            n_pad_clusters = max_n_clusters - n_clusters[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters)
                pad_sample = Variable(pad_sample)
                if self.usegpu:
                    pad_sample = pad_sample.to(device)
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features, max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1) -
                           self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(bs):
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = mean_sample.unsqueeze(2).expand(n_features, n_clusters[i], n_clusters[i])
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            margin = Variable(margin)
            if self.usegpu:
                margin = margin.to(device)
            c_dist = torch.sum(torch.clamp(margin - torch.norm(diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term

from torch.nn.modules.loss import  _Loss

class LogCosh(_Loss):
    def __init__(self, usegpu=True, size_average=True):
        super(LogCosh, self).__init__(size_average)
        self.usegpu = usegpu
    def forward(self, y_hat, y):
        return torch.mean(torch.log(torch.cosh(y-y_hat)))
