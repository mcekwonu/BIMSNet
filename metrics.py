"""
SR: segmentation image
GT: ground truth image
"""
import torch
import torch.nn as nn
import numpy as np


class IOULoss(nn.Module):
    """IOU loss function"""

    def __init__(self):
        super().__init__()

    def forward(self, SR, GT, smooth=1.):
        assert SR.shape == GT.shape, "Predicted and Groundtruth image must have same size!"

        SR = torch.sigmoid(SR)
        SR = SR.view(-1)
        GT = GT.view(-1)

        intersection = (SR * GT).sum() + smooth
        total = (SR + GT).sum()
        union = total - intersection
        iou_loss = 1 - intersection / (union + smooth)

        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        elif self.reduction == 'none':
            return iou_loss
        else:
            raise NotImplementedError(
                "{} reduction method not implemented".format(self.reduction)
            )


class BCELoss(nn.Module):
    """Binary Cross entropy with logit loss"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, SR, GT):
        assert (
                SR.shape == GT.shape
        ), "Predicted and Groundtruth image must have same size!"

        bce_loss = self.bce_loss(SR, GT)

        return bce_loss


def dice_score(SR, GT):
    """Computes the dice coefficient between segmented and ground truth"""
    if SR.shape != GT.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape."
        )

    SR = torch.sigmoid(SR)
    SR = (SR > 0.5).float()

    return 2 * (SR * GT).sum() / ((SR + GT).sum() + 1e-8)


def iou_score(SR, GT):
    """Computes the IOU score"""
    smooth = 1e-8
    SR = (SR > 0.5).float()
    inter = SR * GT
    union = SR + GT

    return inter.sum() / (union.sum() + smooth)


def numeric_score(SR, GT):
    """Computes score:
    FP = False Positives
    FN = False Negative
    TP = True Positives
    TN = True Negative
    return: FP, FN, TP, TN
    """

    FP = ((SR == 1) & (GT == 0)).sum()
    FN = ((SR == 0) & (GT == 1)).sum()
    TP = ((SR == 1) & (GT == 1)).sum()
    TN = ((SR == 0) & (GT == 0)).sum()

    return FP, FN, TP, TN


def accuracy(SR, GT):
    """Getting the accuracy of the model"""

    FP, FN, TP, TN = numeric_score(SR, GT)
    N = FP + FN + TP + TN

    return (TP + TN) / N


def sensitivity(SR, GT):
    """Computes True Positive Rate (TPR)"""
    FP, FN, TP, TN = numeric_score(SR, GT)
    SE = TP / (TP + FN + 1e-8)

    return SE


def specificity(SR, GT):
    """Computes True Negative Rate (TNR)"""

    FP, FN, TP, TN = numeric_score(SR, GT)
    SP = TN / (TN + FP + 1e-8)

    return SP


def precision(SR, GT):
    FP, FN, TP, TN = numeric_score(SR, GT)
    PC = TP / (TP + FP + 1e-8)

    return PC


def F1_score(SR, GT):
    SE = sensitivity(SR, GT)
    PC = precision(SR, GT)

    F1 = 2 * SE * PC / (SE + PC + 1e-8)

    return F1


def JS_score(SR, GT):
    SR = SR > 0.5

    inter = ((SR + GT) == 2).sum()
    union = ((SR + GT) >= 1).sum()
    JS = inter / (union + 1e-8)

    return JS


def DC_score(SR, GT):
    if SR.shape != GT.shape:
        raise ValueError(
            "Shape mismatch: Groundtruth and predicted image must have the same shape."
        )

    SR = (SR > 0.5).float()

    return 2 * (SR * GT).sum() / ((SR + GT).sum() + 1e-8)