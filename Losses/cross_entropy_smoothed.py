'''
Implementation of Cross entropy loss expecting smoothed labels as input.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CrossEntropySmoothed(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=10, size_average=False):
        super(CrossEntropySmoothed, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, input, target):
        '''input and target dimensions should be same
           (say (128, 10) for a minibatch of size 128 and 10 classes'''

        smoothed_target = smooth_one_hot(target, self.num_classes, smoothing=self.smoothing)
        logpt = F.log_softmax(input)
        qlogpt = -(smoothed_target * logpt)
        loss = torch.sum(qlogpt, 1)

        if self.size_average: return loss.mean()
        else: return loss.sum()


def smooth_one_hot(true_labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1).type(torch.LongTensor).to(true_labels.device), confidence)
    return true_dist