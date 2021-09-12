import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import cross_entropy_with_regularizer, focal_loss_with_regularizer
from Losses.loss import focal_loss_with_regularizer_diff_gamma
from Losses.loss import mmce, mmce_weighted
from Losses.loss import brier_score
from Losses.loss import cross_entropy_smoothed
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np
import pdb

def train_cross_entropy(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = cross_entropy(logits, labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_cross_entropy(epoch, embedding_model, model, x_val, y_val, device, data_name, toPrint=True):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += cross_entropy(logits, labels).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    if toPrint:
        print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
            data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]

def test_cross_entropy_ts(epoch, model, x_val, y_val, device, data_name, toPrint=True): #Used for temperature scaling
    model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            logits = model(data)
            loss += cross_entropy(logits, labels).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    if toPrint:
        print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
            data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    return loss / (1+x_val.shape[0]//batch_size)


def train_cross_entropy_smoothed(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size, smoothing=0.0, num_classes=20):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = cross_entropy_smoothed(logits, labels, smoothing=smoothing, num_classes=num_classes)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_cross_entropy_smoothed(epoch, embedding_model, model, x_val, y_val, device, data_name, toPrint=True, smoothing=0.0, num_classes=20):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += cross_entropy_smoothed(logits, labels, smoothing=smoothing, num_classes=num_classes).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    if toPrint:
        print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
            data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]


def test_cross_entropy_ts_smoothed(epoch, model, x_val, y_val, device, data_name, toPrint=True, smoothing=0.0, num_classes=20): #Used for temperature scaling
    model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            logits = model(data)
            loss += cross_entropy_smoothed(logits, labels, smoothing=smoothing, num_classes=num_classes).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    if toPrint:
        print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
            data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    return loss / (1+x_val.shape[0]//batch_size)



def train_mmce_weighted(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size, lamda):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = mmce_weighted(logits, labels, device=device, lamda=lamda)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_mmce_weighted(epoch, embedding_model, model, x_val, y_val, device, data_name, lamda):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += mmce_weighted(logits, labels, device=device, lamda=lamda).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
        data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]


def train_focal_loss(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size, gamma):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = focal_loss(logits, labels, gamma=gamma)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_focal_loss(epoch, embedding_model, model, x_val, y_val, device, data_name, gamma):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += focal_loss(logits, labels, gamma=gamma).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
        data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]


def train_focal_loss_adaptive(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size, gamma):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = focal_loss_adaptive(logits, labels, gamma=gamma, device=device)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_focal_loss_adaptive(epoch, embedding_model, model, x_val, y_val, device, data_name, gamma):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += focal_loss_adaptive(logits, labels, gamma=gamma, device=device).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
        data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]

def train_brier_score(epoch, embedding_model, model, permutation_train, permutation_labels, optimizer, device, batch_size):
    log_interval = 10
    model.train()
    embedding_model.eval()
    train_loss = 0
    overall_acc = 0
    for i in range(permutation_train.shape[0]//batch_size):
        data = torch.from_numpy(permutation_train[i*batch_size:(i+1)*batch_size]).type(torch.LongTensor).to(device)
        labels = torch.from_numpy(np.argmax(permutation_labels[i*batch_size:(i+1)*batch_size], 1)).to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            emb = embedding_model(data)
        
        logits = model(emb)
        loss = brier_score(logits, labels)

        loss.backward()
        #torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        predictions = torch.argmax(logits, 1)
        acc_train = torch.sum(torch.where(torch.eq(predictions, labels),
                    torch.ones(predictions.shape).to(device),
                    torch.zeros(predictions.shape).to(device)))
        overall_acc += acc_train
        optimizer.step()

        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size * (permutation_train.shape[0]//batch_size),
                100. * i / (permutation_train.shape[0]//batch_size),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}, Train Acc: {:.4f}'.format(
        epoch, train_loss / (permutation_train.shape[0]//batch_size), overall_acc/(batch_size * (permutation_train.shape[0]//batch_size))))
    return train_loss / (permutation_train.shape[0]//batch_size)


def test_brier_score(epoch, embedding_model, model, x_val, y_val, device, data_name, toPrint=True):
    model.eval()
    embedding_model.eval()
    loss = 0
    overall_acc = 0
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)
            emb = embedding_model(data)
            logits = model(emb)
            loss += brier_score(logits, labels).item()
            predictions = torch.argmax(logits, 1)
            acc = torch.sum(torch.where(torch.eq(predictions, labels),
                        torch.ones(predictions.shape).to(device),
                        torch.zeros(predictions.shape).to(device)))
            overall_acc += acc

    if toPrint:
        print('======> {:s} set loss: {:.4f}, Acc: {:.4f}'.format(
            data_name, loss/(1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]))
    
    return loss / (1+x_val.shape[0]//batch_size), overall_acc/x_val.shape[0]

def test_classification_net(embedding_model, model, x_val, y_val, device):
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    embedding_model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)

            emb = embedding_model(data)
            logits = model(emb)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())

    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list

def test_classification_net_ts(model, x_val, y_val, device): #Used for temperature scaling
    '''
    This function reports classification accuracy and confusion matrix over a dataset.
    '''
    model.eval()
    labels_list = []
    predictions_list = []
    confidence_vals_list = []
    batch_size = 128
    with torch.no_grad():
        for i in range(1+x_val.shape[0]//batch_size):
            data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).to(device)
            labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).to(device)

            logits = model(data)
            softmax = F.softmax(logits, dim=1)
            confidence_vals, predictions = torch.max(softmax, dim=1)

            labels_list.extend(labels.cpu().numpy().tolist())
            predictions_list.extend(predictions.cpu().numpy().tolist())
            confidence_vals_list.extend(confidence_vals.cpu().numpy().tolist())

    accuracy = accuracy_score(labels_list, predictions_list)
    return confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
        predictions_list, confidence_vals_list
