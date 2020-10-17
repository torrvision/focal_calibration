'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + MMCE_weighted
4. Cross Entropy + MMCE
'''

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.entropy_regularizer import EntropyRegularizer
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore


def cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets, reduction='sum')


def focal_loss(logits, targets, gamma=1):
    return FocalLoss(gamma=gamma)(logits, targets)


def focal_loss_adaptive(logits, targets, gamma=3.0, device=None):
    return FocalLossAdaptive(gamma=gamma,
                             device=device)(logits, targets)


def mmce(logits, targets, device, lamda=1):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(device)(logits, targets)
    return ce + (lamda * mmce)


def mmce_weighted(logits, targets, device, lamda=1):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(device)(logits, targets)
    return ce + (lamda * mmce)


def brier_score(logits, targets):
    return BrierScore()(logits, targets)