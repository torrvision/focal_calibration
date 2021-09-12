'''
This module contains methods for testing models on the val or the test sets using different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import cross_entropy_with_regularizer, focal_loss_with_regularizer
from Losses.loss import focal_loss_with_regularizer_diff_gamma
from Losses.loss import mmce, mmce_weighted
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix
import pdb

def test_cross_entropy(epoch, model, test_val_loader, device):
    '''
    Util method for testing with cross entropy loss.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += cross_entropy(logits, labels).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_focal_loss(epoch, model, test_val_loader, device, gamma=1.0):
    '''
    Util method for testing with focal loss.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += focal_loss(logits, labels, gamma=gamma).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_focal_loss_adaptive(epoch, model, test_val_loader, device, gamma=3.0):
    '''
    Util method for testing with focal loss with adaptive gamma.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += focal_loss_adaptive(logits, labels, gamma=gamma, device=device).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_cross_entropy_with_reg(epoch, model, test_val_loader, device, gamma=1.0, lamda=1.0):
    '''
    Util method for testing with cross entropy with entropy regularizer.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += cross_entropy_with_regularizer(logits, labels, gamma=gamma, lamda=lamda).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_focal_loss_with_reg(epoch, model, test_val_loader, device, gamma=1.0, lamda=1.0):
    '''
    Util method for testing with focal loss with entropy regularizer.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += focal_loss_with_regularizer(logits, labels, gamma=gamma, lamda=lamda).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_focal_loss_with_reg_diff_gamma(epoch, model, test_val_loader, device, gamma1=1.0, gamma2=1.0, lamda=1.0):
    '''
    Util method for testing with focal loss with entropy regularizer with different gammas.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += focal_loss_with_regularizer_diff_gamma(logits, labels, gamma1=gamma1, gamma2=gamma2, lamda=lamda).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_mmce_weighted(epoch, model, test_val_loader, device, lamda=1.0):
    '''
    Util method for testing with mmce_weighted.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += mmce_weighted(logits, labels, device=device, lamda=lamda).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_mmce(epoch, model, test_val_loader, device, lamda=1.0):
    '''
    Util method for testing with mmce.
    '''
    model.eval()
    loss = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_val_loader):
            data = data.to(device)
            labels = labels.to(device)

            logits = model(data)
            loss += mmce(logits, labels, device=device, lamda=lamda).item()
            num_samples += len(data)

    print('======> Test set loss: {:.4f}'.format(
        loss / num_samples))
    return loss / num_samples



def test_classification_net(model, data_loader, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)

            labels_list.extend(label.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())
    accuracy = accuracy_score(labels_list, predictions_list)

    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list

def test_classification_net_pred_avg(model, data_loader, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    p_correct = 0
    p_incorrect = 0
    n_correct = 0
    n_incorrect = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            label = label.view(-1,1)
            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            p_t = softmax.gather(1,label) 

            confidence_vals, predictions = torch.max(softmax, dim=1)
            predictions = predictions.view(-1,1)
            confidence_vals = confidence_vals.view(-1,1)
            correct_mask = torch.where(torch.eq(predictions, label),
                                       torch.ones(predictions.shape).to(device),
                                       torch.zeros(predictions.shape).to(device))
            # pdb.set_trace()
            # p_correct += (correct_mask*p_t).sum().item()
            # p_incorrect += ((1-correct_mask)*p_t).sum().item()
            # n_correct += correct_mask.sum().item()
            # n_incorrect += (1-correct_mask).sum().item()
            p_correct += (correct_mask*confidence_vals).sum().item()
            p_incorrect += ((1-correct_mask)*confidence_vals).sum().item()
            n_correct += correct_mask.sum().item()
            n_incorrect += (1-correct_mask).sum().item()
   
    return p_correct/n_correct, p_incorrect/n_incorrect


def test_classification_net_pred_dist(model, data_loader, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list_correct = []
    confidence_vals_list_incorrect = []
    p_correct = 0
    p_incorrect = 0
    n_correct = 0
    n_incorrect = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            label = label.view(-1,1)
            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            p_t = softmax.gather(1,label) 

            confidence_vals, predictions = torch.max(softmax, dim=1)
            predictions = predictions.view(-1,1)
            confidence_vals = confidence_vals.view(-1,1)
            correct_mask = torch.where(torch.eq(predictions, label),
                                       torch.ones(predictions.shape).to(device),
                                       torch.zeros(predictions.shape).to(device))
           
            
            labels_list.extend(label.view(-1).cpu().numpy().tolist()) 
            predictions_list.extend(predictions.view(-1).cpu().numpy().tolist())

            confidence_vals_list_correct.extend(confidence_vals[correct_mask.view(-1).byte()].view(-1).cpu().numpy().tolist()) 
            confidence_vals_list_incorrect.extend(confidence_vals[(1-correct_mask).view(-1).byte()].view(-1).cpu().numpy().tolist()) 
    

    # pdb.set_trace()
    # print(((np.array(confidence_vals_list_correct)>=0.99).sum()+(np.array(confidence_vals_list_incorrect)>=0.99).sum())/(np.array(confidence_vals_list_correct).sum()+np.array(confidence_vals_list_incorrect).sum()))
    return confidence_vals_list_correct, confidence_vals_list_incorrect, labels_list, predictions_list


# |S99| scores (which contradict Table 2 results, mostly because models aren't the same?):
# 0.10319942908801868 for fg_3 with temp
# 0.41523550837184475 for fg_3 without temp
#
# As the above didn't match, I didn't add the following to supplementary:
# To further investigate how networks trained with Focal loss have optimal temperature close to one, we tried finding the average $\hat{p}_{i,\hat{y}_i}$, which is the model confidence for the predicted class, for the models trained with the cross-entropy and focal loss, before and after temperature tuning. The result in Table \ref{table:avg_p_table} shows that the models trained with focal loss already have very low confidence on incorrect predictions, similar to cross-entropy after temperature scaling.
# %---
# \begin{table*}[!t]
# \centering
# \scriptsize
# \resizebox{\linewidth}{!}{%
# \begin{tabular}{ccccccccccc}
# \toprule
# \textbf{Dataset} & \textbf{Model} & \multicolumn{2}{c}{\textbf{Cross Entropy (Pre T)}} & \multicolumn{2}{c}{\textbf{Cross Entropy (Post T)}} &  \multicolumn{2}{c}{\textbf{Focal Loss (Pre T)}} & \multicolumn{2}{c}{\textbf{Focal Loss (Post T)}} \\
# && avg $\hat{p}_{i,\hat{y}_i}$ cor & avg $\hat{p}_{i,\hat{y}_i}$ inc & avg $\hat{p}_{i,\hat{y}_i}$ cor & avg $\hat{p}_{i,\hat{y}_i}$ inc & avg $\hat{p}_{i,\hat{y}_i}$ cor & avg $\hat{p}_{i,\hat{y}_i}$ inc & avg $\hat{p}_{i,\hat{y}_i}$ cor & avg $\hat{p}_{i,\hat{y}_i}$ inc \\
# \midrule
# CIFAR-10 & ResNet 50 & $99.68$ & $93.70$ & $97.17$ & $80.97$ & $97.06$ & $77.82$ & $96.04$ & $75.58$ \\ 
# \bottomrule
# \end{tabular}}
# \caption{Percentage of test samples predicted with confidence higher than $99\%$ and the corresponding accuracy for Cross Entropy, MMCE and Focal loss computed both pre and post temperature scaling (represented in the table as pre T and post T respectively).}
# \label{table:avg_p_table}
# %\vspace{-2\baselineskip}
# \end{table*}
# % ---
