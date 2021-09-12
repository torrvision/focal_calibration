from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from NewsGroup.ng_model import GlobalPoolingCNN
from NewsGroup.ng_utils import train_cross_entropy, test_cross_entropy
from NewsGroup.ng_utils import train_cross_entropy_smoothed, test_cross_entropy_smoothed
from NewsGroup.ng_utils import train_mmce_weighted, test_mmce_weighted
from NewsGroup.ng_utils import train_focal_loss, test_focal_loss
from NewsGroup.ng_utils import train_focal_loss_adaptive, test_focal_loss_adaptive
from NewsGroup.ng_utils import train_brier_score, test_brier_score
from NewsGroup.ng_loader import ng_loader

import argparse
import json
import pdb

def parseArgs():

    parser = argparse.ArgumentParser(
        description="Training for 20 Newsgroups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lamda", type=float, default=8.0,
                    dest="lamda", help='Coefficient for MMCE error term.')
    parser.add_argument("--batch_size", type=int, default=128, dest="batch_size",
                        help='Batch size for training.')
    parser.add_argument("--num_epochs", type=int, default=22, dest="num_epochs",
                        help='Number of epochs of training.')
    parser.add_argument("--save-path", type=str, default='./',
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--loss", type=str, default='cross_entropy', dest="loss_function",
                        help="Loss function to be used for training")
    parser.add_argument("--gamma", type=float, default=3.0,
                        dest="gamma", help="Gamma for focal components")
    parser.add_argument("--gamma2", type=float, default=1.0,
                        dest="gamma2", help="Gamma for different focal components")
    parser.add_argument("--gamma3", type=float, default=1.0,
                        dest="gamma3", help="Gamma for different focal components")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        dest="smoothing", help="Smoothing factor for labels")
    parser.add_argument("--gamma-schedule", type=int, default=0,
        dest="gamma_schedule", help="Schedule gamma or not")

    return parser.parse_args()


if __name__ == "__main__":

    args = parseArgs()

    torch.manual_seed(1)
    device = torch.device("cuda")

    embedding_matrix, x_train, y_train, x_pval, y_pval, x_val, y_val, num_words, EMBEDDING_DIM = ng_loader()
    embedding_model = nn.Embedding(num_words, EMBEDDING_DIM)

    net=GlobalPoolingCNN(keep_prob=0.7)
    embedding_matrix.cuda()
    embedding_model.cuda()
    net.cuda()

    embedding_model.state_dict()['weight'].copy_(embedding_matrix)

    opt_params = net.parameters()
    optimizer = optim.Adam(opt_params)

    batch_size = args.batch_size
    num_epochs = args.num_epochs

    training_set_loss = {}
    val_set_loss = {}
    test_set_loss = {}

    max_val = 0

    if args.loss_function == 'cross_entropy':
        for epoch in range(0, num_epochs):
            perm = np.random.permutation(np.arange(len(x_train)))
            permutation_train = np.take(x_train, perm, axis=0)
            permutation_labels = np.take(y_train, perm, axis=0)

            train_loss = train_cross_entropy(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size)
            val_loss, val_acc = test_cross_entropy(epoch, embedding_model, net, x_pval, y_pval, device, 'Val')
            test_loss, _ = test_cross_entropy(epoch, embedding_model, net, x_val, y_val, device, 'Test')
            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss

            if max_val < val_acc:
                max_val = val_acc
                save_name = args.save_loc + args.loss_function + '_best'
                torch.save(net.state_dict(), save_name + '.model')

        save_name = args.save_loc + args.loss_function + '_' + str(epoch + 1)
        torch.save(net.state_dict(), save_name + '.model')


    if args.loss_function == 'cross_entropy_smoothed':
        for epoch in range(0, num_epochs):
            perm = np.random.permutation(np.arange(len(x_train)))
            permutation_train = np.take(x_train, perm, axis=0)
            permutation_labels = np.take(y_train, perm, axis=0)

            train_loss = train_cross_entropy_smoothed(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size, smoothing=args.smoothing, num_classes=20)
            val_loss, val_acc = test_cross_entropy_smoothed(epoch, embedding_model, net, x_pval, y_pval, device, 'Val', smoothing=args.smoothing, num_classes=20)
            test_loss, _ = test_cross_entropy_smoothed(epoch, embedding_model, net, x_val, y_val, device, 'Test', smoothing=args.smoothing, num_classes=20)
            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss

            if max_val < val_acc:
                max_val = val_acc
                save_name = args.save_loc + args.loss_function + '_smoothing_' + str(args.smoothing) + '_best'
                torch.save(net.state_dict(), save_name + '.model')

        save_name = args.save_loc + args.loss_function + '_smoothing_' + str(args.smoothing) + '_' + str(epoch + 1)
        torch.save(net.state_dict(), save_name + '.model')



    if args.loss_function == 'mmce_weighted':
        for epoch in range(0, num_epochs):
            perm = np.random.permutation(np.arange(len(x_train)))
            permutation_train = np.take(x_train, perm, axis=0)
            permutation_labels = np.take(y_train, perm, axis=0)

            train_loss = train_mmce_weighted(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size, lamda=args.lamda)
            val_loss, val_acc = test_mmce_weighted(epoch, embedding_model, net, x_pval, y_pval, device, 'Val', lamda=args.lamda)
            test_loss, _ = test_mmce_weighted(epoch, embedding_model, net, x_val, y_val, device, 'Test', lamda=args.lamda)
            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss

            if max_val < val_acc:
                max_val = val_acc
                save_name = args.save_loc + args.loss_function + '_lamda_' + str(args.lamda) + '_best'
                torch.save(net.state_dict(), save_name + '.model')

        save_name = args.save_loc + args.loss_function + '_lamda_' + str(args.lamda) + '_' + str(epoch + 1)
        torch.save(net.state_dict(), save_name + '.model')

    if args.loss_function == 'focal_loss':
        if (args.gamma_schedule == 1):  #TODO: You can change the scheduling thresholds if num_epochs changes
            for epoch in range(0, num_epochs):
                if (epoch < int(num_epochs*100/350)):
                    gamma = args.gamma
                elif (epoch >= int(num_epochs*100/350) and epoch < int(num_epochs*250/350)):
                    gamma = args.gamma2
                else:
                    gamma = args.gamma3

                perm = np.random.permutation(np.arange(len(x_train)))
                permutation_train = np.take(x_train, perm, axis=0)
                permutation_labels = np.take(y_train, perm, axis=0)

                train_loss = train_focal_loss(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size, gamma=gamma)
                val_loss, val_acc = test_focal_loss(epoch, embedding_model, net, x_pval, y_pval, device, 'Val', gamma=gamma)
                test_loss, _ = test_focal_loss(epoch, embedding_model, net, x_val, y_val, device, 'Test', gamma=gamma)
                training_set_loss[epoch] = train_loss
                val_set_loss[epoch] = val_loss
                test_set_loss[epoch] = test_loss
            
                if max_val < val_acc:
                    max_val = val_acc
                    save_name = args.save_loc + args.loss_function + '_scheduled_gamma_' + str(args.gamma) + '_' + str(args.gamma2) + '_' + str(args.gamma3) + '_best'
                    torch.save(net.state_dict(), save_name + '.model')

            save_name = args.save_loc + args.loss_function + '_scheduled_gamma_' + str(args.gamma) + '_' + str(args.gamma2) + '_' + str(args.gamma3) + '_' + str(epoch + 1)
            torch.save(net.state_dict(), save_name + '.model')
        else:
            for epoch in range(0, num_epochs):
                perm = np.random.permutation(np.arange(len(x_train)))
                permutation_train = np.take(x_train, perm, axis=0)
                permutation_labels = np.take(y_train, perm, axis=0)

                train_loss = train_focal_loss(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size, gamma=args.gamma)
                val_loss, val_acc = test_focal_loss(epoch, embedding_model, net, x_pval, y_pval, device, 'Val', gamma=args.gamma)
                test_loss, _ = test_focal_loss(epoch, embedding_model, net, x_val, y_val, device, 'Test', gamma=args.gamma)
                training_set_loss[epoch] = train_loss
                val_set_loss[epoch] = val_loss
                test_set_loss[epoch] = test_loss

                if max_val < val_acc:
                    max_val = val_acc
                    save_name = args.save_loc + args.loss_function + '_gamma_' + str(args.gamma) + '_best'
                    torch.save(net.state_dict(), save_name + '.model')


            save_name = args.save_loc + args.loss_function + '_gamma_' + str(args.gamma) + '_' + str(epoch + 1)
            torch.save(net.state_dict(), save_name + '.model')

    if args.loss_function == 'focal_loss_adaptive':
        for epoch in range(0, num_epochs):
            perm = np.random.permutation(np.arange(len(x_train)))
            permutation_train = np.take(x_train, perm, axis=0)
            permutation_labels = np.take(y_train, perm, axis=0)

            train_loss = train_focal_loss_adaptive(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size, gamma=args.gamma)
            val_loss, val_acc = test_focal_loss_adaptive(epoch, embedding_model, net, x_pval, y_pval, device, 'Val', gamma=args.gamma)
            test_loss, _ = test_focal_loss_adaptive(epoch, embedding_model, net, x_val, y_val, device, 'Test', gamma=args.gamma)

            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss
            
            if max_val < val_acc:
                max_val = val_acc
                save_name = args.save_loc + args.loss_function + '_gamma_' + str(args.gamma) + '_best'
                torch.save(net.state_dict(), save_name + '.model')

        save_name = args.save_loc + args.loss_function + '_gamma_' + str(args.gamma) + '_' + str(epoch + 1)
        torch.save(net.state_dict(), save_name + '.model')
    
    if args.loss_function == 'brier_score':
        for epoch in range(0, num_epochs):
            perm = np.random.permutation(np.arange(len(x_train)))
            permutation_train = np.take(x_train, perm, axis=0)
            permutation_labels = np.take(y_train, perm, axis=0)

            train_loss = train_brier_score(epoch, embedding_model, net, permutation_train, permutation_labels, optimizer, device, batch_size)
            val_loss, val_acc = test_brier_score(epoch, embedding_model, net, x_pval, y_pval, device, 'Val')
            test_loss, _ = test_brier_score(epoch, embedding_model, net, x_val, y_val, device, 'Test')
            training_set_loss[epoch] = train_loss
            val_set_loss[epoch] = val_loss
            test_set_loss[epoch] = test_loss

            if max_val < val_acc:
                max_val = val_acc
                save_name = args.save_loc + args.loss_function + '_best'
                torch.save(net.state_dict(), save_name + '.model')

        save_name = args.save_loc + args.loss_function + '_' + str(epoch + 1)
        torch.save(net.state_dict(), save_name + '.model')


    with open(save_name + '_train_loss.json', 'w') as f:
        json.dump(training_set_loss, f)

    with open(save_name + '_val_loss.json', 'w') as fv:
        json.dump(val_set_loss, fv)

    with open(save_name + '_test_loss.json', 'w') as ft:
        json.dump(test_set_loss, ft)
