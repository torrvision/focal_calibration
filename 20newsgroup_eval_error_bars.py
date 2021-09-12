import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import torch
import torch.nn as nn
from torch.nn import functional as F

import random
import matplotlib.pyplot as plt

from NewsGroup.ng_model import GlobalPoolingCNN
from NewsGroup.ng_utils import test_cross_entropy, test_classification_net
from NewsGroup.ng_utils import test_cross_entropy_ts, test_classification_net_ts
from NewsGroup.ng_loader import ng_loader
from Experiments.temperature_scaling import ModelWithTemperature_ng
from Experiments.error_bars import get_calibration_error_uncertainties

# Import metrics to compute
from Metrics.metrics import expected_calibration_error
from Metrics.metrics import maximum_calibration_error
from Metrics.plots import reliability_plot, bin_strength_plot


import pdb
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Obtaining calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_epochs", type=int, default=22, dest="num_epochs",
                        help='Number of epochs of training.')
    parser.add_argument("--save-path", type=str, default='./',
                        dest="save_loc",
                        help='Path to export the model')
    parser.add_argument("--num-bins", type=int, default=15, dest="num_bins",  
                        help='Number of bins')
    parser.add_argument('--saved_model_names', nargs='+', default='resnet110_mmce_weighted_lamda_4.0_250.model',
                        help='list of file names of the pre-trained model')
    parser.add_argument("--temp", type=float, default=1.0, dest="temp", help="Optimal temperature")
    parser.add_argument("-cn", action="store_true", dest="cross_validate_on_nll",
                        help="cross validate on nll")
    parser.set_defaults(cross_validate_on_nll=False)
    parser.add_argument("-ce", action="store_true", dest="cross_validate_on_ece",
                        help="cross validate on ece")
    parser.set_defaults(cross_validate_on_ece=False)
    parser.add_argument("-se", action="store_true", dest="cross_validate_on_sce",
                        help="cross validate on sce")
    parser.set_defaults(cross_validate_on_sce=False)
    parser.add_argument("-cae", action="store_true", dest="cross_validate_on_adaece",
                        help="cross validate on adaptive ece")
    parser.set_defaults(cross_validate_on_adaece=False)
    parser.add_argument("-tn", action="store_true", dest="train_on_nll",
                        help="train on nll")
    parser.set_defaults(train_on_nll=False)
    parser.add_argument("-ta", action="store_true", dest="train_all",
                        help="evaluate all the metrices")
    parser.set_defaults(train_all=False)
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to log")
    parser.set_defaults(log=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    torch.manual_seed(1)
    device = torch.device("cuda")    

    num_bins = args.num_bins
    opt_temp = args.temp

    # Taking input for the dataset
    embedding_matrix, x_train, y_train, x_pval, y_pval, x_val, y_val, num_words, EMBEDDING_DIM = ng_loader()
    embedding_model = nn.Embedding(num_words, EMBEDDING_DIM)
    embedding_matrix.cuda()
    embedding_model.cuda()
    embedding_model.state_dict()['weight'].copy_(embedding_matrix)

    for saved_model_name in args.saved_model_names:
        # Evaluating the model at T = 1
        # Getting the number of bins
        net = GlobalPoolingCNN(keep_prob=0.7, temp=1.0)
        net.cuda()
        net.load_state_dict(torch.load(args.save_loc + saved_model_name))

        logits_list = []
        labels_list = []
        batch_size = 128
        embedding_model.eval()
        net.eval()
        with torch.no_grad():
            for i in range(1+x_val.shape[0]//batch_size):
                data = torch.from_numpy(x_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])]).type(torch.LongTensor).cuda()
                labels = torch.from_numpy(np.argmax(y_val[i*batch_size:min((i+1)*batch_size, x_val.shape[0])], 1)).cuda()
                emb = embedding_model(data)
                logits = net(emb)
                logits_list.append(logits)
                labels_list.append(labels)

            all_logits = torch.cat(logits_list).cuda()
            all_labels = torch.cat(labels_list).cuda()

            all_temp_scaled_logits = all_logits / opt_temp
            probs = F.softmax(all_logits, dim=1)
            temp_scaled_probs = F.softmax(all_temp_scaled_logits, dim=1)
        
            # Pre temp scaling ECE, Ada-ECE and SCE
            [ece_lower, ece_mid, ece_upper] = get_calibration_error_uncertainties(probs, all_labels, 'ece')
            [aece_lower, aece_mid, aece_upper] = get_calibration_error_uncertainties(probs, all_labels, 'aece')
            [sce_lower, sce_mid, sce_upper] = get_calibration_error_uncertainties(probs, all_labels, 'sce')
        
            # Post temp scaling ECE, Ada-ECE and SCE
            [ece_t_lower, ece_t_mid, ece_t_upper] = get_calibration_error_uncertainties(temp_scaled_probs, all_labels, 'ece')
            [aece_t_lower, aece_t_mid, aece_t_upper] = get_calibration_error_uncertainties(temp_scaled_probs, all_labels, 'aece')
            [sce_t_lower, sce_t_mid, sce_t_upper] = get_calibration_error_uncertainties(temp_scaled_probs, all_labels, 'sce')
        
            # Test NLL & ECE & MCE
            print(saved_model_name)
            print('=====================================================================================')
            print ('Pre temp scaling: ')
            print ('ECE_lower: ' + str(ece_lower) + ' Mid: ' + str(ece_mid) + ' Upper: ' + str(ece_upper))
            print ('AECE_lower: ' + str(aece_lower) + ' Mid: ' + str(aece_mid) + ' Upper: ' + str(aece_upper))
            print ('SCE_lower: ' + str(sce_lower) + ' Mid: ' + str(sce_mid) + ' Upper: ' + str(sce_upper))
            print('=====================================================================================')
            print ('Post temp scaling: ')
            print ('ECE_lower: ' + str(ece_t_lower) + ' Mid: ' + str(ece_t_mid) + ' Upper: ' + str(ece_t_upper))
            print ('AECE_lower: ' + str(aece_t_lower) + ' Mid: ' + str(aece_t_mid) + ' Upper: ' + str(aece_t_upper))
            print ('SCE_lower: ' + str(sce_t_lower) + ' Mid: ' + str(sce_t_mid) + ' Upper: ' + str(sce_t_upper))
            print('=====================================================================================')
