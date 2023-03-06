import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class mIoUMask(torch.nn.Module):

    def __init__(self, num_classes=3, ignore_class=None, eps=1e-7):
        super(mIoUMask, self).__init__()
        self.eps = eps
        self.num_class = num_classes + (1 if ignore_class is not None else 0)
        self.ignore_class = ignore_class
        self.confusion_matrix = np.zeros((self.num_class, )*2)

    def _generate_matrix(self, pre_image, gt_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) 
        if self.ignore_class is not None:
            mask = mask & (gt_image != self.ignore_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def Tissue_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU[np.isnan(MIoU)] = 0
        return MIoU

    def Mean_Intersection_over_Union(self):
        MIoU = self.Tissue_Intersection_over_Union()
        MIoU = np.mean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def forward(self, logits, mask, probs=False):
        if probs:
            pred = torch.argmax(logits, dim=1).byte()
        else:
            pred = F.softmax(logits, dim=1)
            pred = torch.argmax(pred, dim=1).byte()  # [N, H, W]

        mask = mask.byte()  # [N, H, W]
        # [N, H, W] -> [N, 1, H, W]  1 for background, 0 for foreground

        self.add_batch(pred.cpu().numpy(), mask.cpu().numpy())

        return self.Mean_Intersection_over_Union(), self.Frequency_Weighted_Intersection_over_Union()


class DiceLoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets, bkg):

        B, C, H, W = inputs.shape

        # [N, C, H, W]
        assert torch.isnan(inputs).any() == False, inputs

        # [N, H, W] -> [N, H, W, C] -> [N, C, H, W]
        targets = F.one_hot(targets.long(), num_classes=4).byte().permute(
            0, 3, 1, 2)  # label 3: background
        targets = targets.byte()  # one hot

        # [N, H, W] -> [N, 1, H, W]
        bkg = targets[:, -1, :, :].unsqueeze(1)
        bkg = 1 - bkg

        inputs = inputs * bkg
        targets = targets * bkg

        dices = []
        nb_lbls = []

        for c in range(C):
            inputs_c, targets_c = inputs[:, c, :, :], targets[:, c, :, :]

            # [N, H, W] -> [N * H * W]
            inputs_c = inputs_c.reshape(-1)
            targets_c = targets_c.reshape(-1)

            intersection_c = (inputs_c * targets_c).sum()

            dice_c = (2. * intersection_c) / \
                (inputs_c.sum() + targets_c.sum() + self.eps)

            dices.append(dice_c)
            nb_lbls.append(targets_c.sum() + inputs_c.sum())

        dice = sum(dices) / len(dices)

        return 1 - dice
