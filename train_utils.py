'''
This module contains methods for training models with different loss functions.
'''

import torch
from torch.nn import functional as F
from torch import nn

from Losses.loss import cross_entropy, focal_loss, focal_loss_adaptive
from Losses.loss import cross_entropy_with_regularizer, focal_loss_with_regularizer
from Losses.loss import focal_loss_with_regularizer_diff_gamma
from Losses.loss import mmce, mmce_weighted


def train_cross_entropy(epoch, model, train_loader, optimizer, device):
    '''
    Util method for training with cross entropy loss.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_focal_loss(epoch, model, train_loader, optimizer, device, gamma=1.0):
    '''
    Util method for training with focal loss.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = focal_loss(logits, labels, gamma=gamma)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_focal_loss_adaptive(epoch, model, train_loader, optimizer, device, gamma=3.0):
    '''
    Util method for training with focal loss with adaptive gamma.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = focal_loss_adaptive(logits, labels, gamma=gamma, device=device)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_cross_entropy_with_reg(epoch, model, train_loader, optimizer, device, gamma=1.0, lamda=1.0):
    '''
    Util method for training with cross entropy with entropy regularizer.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = cross_entropy_with_regularizer(logits, labels, gamma=gamma, lamda=lamda)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_focal_loss_with_reg(epoch, model, train_loader, optimizer, device, gamma=1.0, lamda=1.0):
    '''
    Util method for training with focal loss with entropy regularizer.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = focal_loss_with_regularizer(logits, labels, gamma=gamma, lamda=lamda)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_focal_loss_with_reg_diff_gamma(epoch, model, train_loader, optimizer, device, gamma1=1.0, gamma2=1.0, lamda=1.0):
    '''
    Util method for training with focal loss with entropy regularizer with different gammas.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = focal_loss_with_regularizer_diff_gamma(logits, labels, gamma1=gamma1, gamma2=gamma2, lamda=lamda)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_mmce_weighted(epoch, model, train_loader, optimizer, device, lamda=1.0):
    '''
    Util method for training with mmce_weighted.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = mmce_weighted(logits, labels, device=device, lamda=lamda)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples



def train_mmce(epoch, model, train_loader, optimizer, device, lamda=1.0):
    '''
    Util method for training with mmce.
    '''
    log_interval = 10
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    num_samples = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)
        loss = mmce(logits, labels, device=device, lamda=lamda)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        train_loss += loss.item()
        optimizer.step()
        num_samples += len(data)

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader),
                loss.item()))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / num_samples))
    return train_loss / num_samples