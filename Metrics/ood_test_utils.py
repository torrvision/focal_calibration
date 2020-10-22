# Utility functions to get OOD detection ROC curves and AUROC scores
# Ideally should be agnostic of model architectures

import torch
import torch.nn.functional as F
from sklearn import metrics


def entropy(net_output):
    p = F.softmax(net_output, dim=1)
    logp = F.log_softmax(net_output, dim=1)
    plogp = p * logp
    entropy = - torch.sum(plogp, dim=1)
    return entropy

def confidence(net_output):
    p = F.softmax(net_output, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence


def get_roc_auc(net, test_loader, ood_test_loader, device):
    bin_labels_entropies = None
    bin_labels_confidences = None
    entropies = None
    confidences = None

    net.eval()
    with torch.no_grad():
        # Getting entropies for in-distribution data
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            bin_label_entropy = torch.zeros(label.shape).to(device)
            bin_label_confidence = torch.ones(label.shape).to(device)

            net_output = net(data)

            entrop = entropy(net_output)
            conf = confidence(net_output)

            if (i == 0):
                bin_labels_entropies = bin_label_entropy
                bin_labels_confidences = bin_label_confidence
                entropies = entrop
                confidences = conf
            else:
                bin_labels_entropies = torch.cat((bin_labels_entropies, bin_label_entropy))
                bin_labels_confidences = torch.cat((bin_labels_confidences, bin_label_confidence))
                entropies = torch.cat((entropies, entrop))
                confidences = torch.cat((confidences, conf))

        # Getting entropies for OOD data
        for i, (data, label) in enumerate(ood_test_loader):
            data = data.to(device)
            label = label.to(device)

            bin_label_entropy = torch.ones(label.shape).to(device)
            bin_label_confidence = torch.zeros(label.shape).to(device)

            net_output = net(data)
            entrop = entropy(net_output)
            conf = confidence(net_output)

            bin_labels_entropies = torch.cat((bin_labels_entropies, bin_label_entropy))
            bin_labels_confidences = torch.cat((bin_labels_confidences, bin_label_confidence))
            entropies = torch.cat((entropies, entrop))
            confidences = torch.cat((confidences, conf))

    fpr_entropy, tpr_entropy, thresholds_entropy = metrics.roc_curve(bin_labels_entropies.cpu().numpy(), entropies.cpu().numpy())
    fpr_confidence, tpr_confidence, thresholds_confidence = metrics.roc_curve(bin_labels_confidences.cpu().numpy(), confidences.cpu().numpy())
    auc_entropy = metrics.roc_auc_score(bin_labels_entropies.cpu().numpy(), entropies.cpu().numpy())
    auc_confidence = metrics.roc_auc_score(bin_labels_confidences.cpu().numpy(), confidences.cpu().numpy())

    return (fpr_entropy, tpr_entropy, thresholds_entropy), (fpr_confidence, tpr_confidence, thresholds_confidence), auc_entropy, auc_confidence
