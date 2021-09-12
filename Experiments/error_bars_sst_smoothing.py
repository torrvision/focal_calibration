import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import bisect
import pickle
from sklearn.linear_model import LogisticRegression

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1].cuda()
        self.bin_uppers = bin_boundaries[1:].cuda()

    def forward(self, probs, labels):
        softmaxes = torch.stack(probs)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        labels = torch.stack(labels)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = Variable(torch.zeros(1)).type_as(softmaxes)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[Variable(in_bin.data)].float().mean()
                avg_confidence_in_bin = confidences[Variable(in_bin.data)].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class _AdaptiveECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_AdaptiveECELoss, self).__init__()
        self.nbins = n_bins
    
    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, probs, labels):
        softmaxes = torch.stack(probs)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        labels = torch.stack(labels)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #import pdb; pdb.set_trace()
        n, bin_boundaries = np.histogram(confidences.cpu().data.numpy(), self.histedges_equalN(confidences.cpu().data.numpy()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = Variable(torch.zeros(1)).type_as(softmaxes)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[Variable(in_bin.data)].float().mean()
                avg_confidence_in_bin = confidences[Variable(in_bin.data)].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class _SCELoss(nn.Module):
    """
    Calculates the Static Calibration Error of a model.
    See: Nixon, J.; Dusenberry, M.; Zhang, L.; Jerfel, G.; and Tran, D. 2019.
    Measuring calibration in deep learning. arXiv preprint arXiv:1904.01685
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_SCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, probs, labels):
        softmaxes = torch.stack(probs)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        labels = torch.stack(labels)
        num_classes = int((torch.max(labels) + 1).item())
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = torch.zeros(1, device=softmaxes.device)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = labels_in_class[in_bin].float().mean()
                    avg_confidence_in_bin = class_confidences[in_bin].mean()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce

def ece_loss(data):
    probs, labels = zip(*data)
    return _ECELoss().cuda()(probs, labels).item()

def aece_loss(data):
    probs, labels = zip(*data)
    return _AdaptiveECELoss().cuda()(probs, labels).item()

def sce_loss(data):
    probs, labels = zip(*data)
    return _SCELoss().cuda()(probs, labels).item()

loss_map = {
    'ece': ece_loss,
    'aece': aece_loss,
    'sce': sce_loss
}

def resample(data):
    indices = np.random.choice(list(range(len(data))), size=len(data), replace=True)
    return [data[i] for i in indices]

def bootstrap_uncertainty(data, functional, estimator=None, alpha=10.0,
                          num_samples=1000):
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    if estimator is None:
        estimator = functional
    estimate = estimator(data)
    plugin = functional(data)
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return (plugin + estimate - np.percentile(bootstrap_estimates, 100 - alpha / 2.0),
            plugin + estimate - np.percentile(bootstrap_estimates, 50),
            plugin + estimate - np.percentile(bootstrap_estimates, alpha / 2.0))


def get_calibration_error_uncertainties(probs, labels, ce_functional='ece', num_samples=1000, alpha=10.0):
    """Get confidence intervals for the calibration error.
    Args:
        probs: A numpy array of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A numpy array of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
    Returns:
        [lower, mid, upper]: 1-alpha confidence intervals produced by bootstrap resampling.
        [lower, upper] represents the confidence interval. mid represents the median of
        the bootstrap estimates. When p is not 2 (e.g. for the ECE where p = 1), this
        can be used as a debiased estimate as well.
    """
    data = list(zip(probs, labels))
    [lower, mid, upper] = bootstrap_uncertainty(data, loss_map[ce_functional], num_samples=num_samples, alpha=alpha)
    return [lower, mid, upper]
