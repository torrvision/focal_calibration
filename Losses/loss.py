'''
Implementation of the following loss functions:
1. Cross Entropy
2. Focal Loss
3. Cross Entropy + Entropy Regularizer
4. Focal Loss + Entropy Regularizer
5. Cross Entropy + MMCE_weighted
6. Cross Entropy + MMCE
'''

from torch.nn import functional as F
from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma import FocalLossAdaptive
from Losses.entropy_regularizer import EntropyRegularizer
from Losses.mmce import MMCE, MMCE_weighted
from Losses.brier_score import BrierScore
from Losses.cross_entropy_smoothed import CrossEntropySmoothed


def cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets)


def cross_entropy_smoothed(logits, targets, smoothing=0.0, num_classes=10):
    return CrossEntropySmoothed(smoothing=smoothing, num_classes=num_classes)(logits, targets)


def focal_loss(logits, targets, gamma=1):
    return FocalLoss(gamma=gamma)(logits, targets)


def focal_loss_adaptive(logits, targets, gamma=3.0, device=None):
    return FocalLossAdaptive(gamma=gamma,
                             device=device)(logits, targets)


def cross_entropy_with_regularizer(logits, targets, gamma=1, lamda=1):
	ce = F.cross_entropy(logits, targets)
	reg = EntropyRegularizer(gamma=gamma)(logits, targets)
	return ce + (lamda * reg)


def focal_loss_with_regularizer(logits, targets, gamma=1, lamda=1):
	f = FocalLoss(gamma=gamma)(logits, targets)
	reg = EntropyRegularizer(gamma=gamma)(logits, targets)
	return f + (lamda * reg)


def focal_loss_with_regularizer_diff_gamma(logits, targets, gamma1=1, gamma2=1, lamda=1):
	f = FocalLoss(gamma=gamma1)(logits, targets)
	reg = EntropyRegularizer(gamma=gamma2)(logits, targets)
	return f + (lamda * reg)


def mmce(logits, targets, device, lamda=1.0):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE(device)(logits, targets)
    return ce + (lamda * mmce)


def mmce_weighted(logits, targets, device, lamda=1.0):
    ce = F.cross_entropy(logits, targets)
    mmce = MMCE_weighted(device)(logits, targets)
    return ce + (lamda * mmce)

def brier_score(logits, targets):
    return BrierScore()(logits, targets)
