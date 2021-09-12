import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import torch
import torch.nn as nn

import random
import matplotlib.pyplot as plt

from NewsGroup.ng_model import GlobalPoolingCNN
from NewsGroup.ng_utils import test_cross_entropy, test_classification_net
from NewsGroup.ng_utils import test_cross_entropy_ts, test_classification_net_ts
from NewsGroup.ng_loader import ng_loader
from Experiments.temperature_scaling import ModelWithTemperature_ng

# Import metrics to compute
from Metrics.metrics import expected_calibration_error
from Metrics.metrics import maximum_calibration_error
from Metrics.metrics import l2_error
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
        conf_matrix, p_accuracy, labels, predictions, confidences = test_classification_net(embedding_model, net, x_val, y_val, device)
        p_ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        p_mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        p_l2 = l2_error(confidences, predictions, labels, num_bins=num_bins)
        p_nll, _ = test_cross_entropy(1, embedding_model, net, x_val, y_val, device, 'Test')

        # Printing the required evaluation metrics
        #print (conf_matrix)
        # print ('Test error: ' + str((1 - p_accuracy)))
        # print ('Test NLL: ' + str(p_nll))
        # print ('ECE: ' + str(p_ece))
        # print ('MCE: ' + str(p_mce))

        # Plotting the reliability plot
        # reliability_plot(confidences, predictions, labels, num_bins=num_bins)
        # bin_strength_plot(confidences, predictions, labels, num_bins=num_bins)

        if args.cross_validate_on_nll:
            # Tuning the temperature parameter on the val set
            T = 0.1
            nll_val = 10 ** 7
            T_opt = 0.1
            Ts = []
            nlls = []
            for i in range(100):
                net = GlobalPoolingCNN(keep_prob=0.7, temp=T)
                net.cuda()
                net.load_state_dict(torch.load(args.save_loc + saved_model_name))
                nll, _ = test_cross_entropy(1, embedding_model, net, x_pval, y_pval, device, 'Val', toPrint = False)
                # print ('At temperature: ' + str(T) + ' Val set NLL: ' + str(nll))
                if (nll < nll_val):
                    nll_val = nll
                    T_opt = T
                Ts.append(T)
                nlls.append(nll)
                T = T + 0.1
            # print ('Optimal temperature by cross validation on nll: ' + str(T_opt))

            # Plotting change in val set NLL for different
            # plt.figure(figsize=(10,8))
            # plt.rcParams.update({'font.size': 20})
            # plt.plot(Ts, nlls, 'r-')
            # plt.xlabel('Temperature values')
            # plt.ylabel('Val set NLL')
            # plt.show()

            # Evaluating the model at T = T_opt
            # Getting the number of bins
            net = GlobalPoolingCNN(keep_prob=0.7, temp=T_opt)
            net.cuda()
            net.load_state_dict(torch.load(args.save_loc + saved_model_name))

            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net(embedding_model, net, x_val, y_val, device)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            nll, _ = test_cross_entropy(1, embedding_model, net, x_val, y_val, device, 'Test')

        elif args.cross_validate_on_ece:
           # # Tuning the temperature parameter on the val set
           # T = 0.1
           # ece_val = 10 ** 7
           # T_opt = 0.1
           # Ts = []
           # eces = []
           # for i in range(100):
           #     net = GlobalPoolingCNN(keep_prob=0.7, temp=T)
           #     net.cuda()
           #     net.load_state_dict(torch.load(args.save_loc + saved_model_name))
           #     _, _, labels, predictions, confidences = test_classification_net(embedding_model, net, x_pval, y_pval, device)
           #     ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)

           #     # print ('At temperature: ' + str(T) + ' Val set ECE: ' + str(ece))
           #     if (ece < ece_val):
           #         ece_val = ece
           #         T_opt = T
           #     Ts.append(T)
           #     eces.append(ece)
           #     T = T + 0.1
           # # print ('Optimal temperature by cross validation on ece: ' + str(T_opt))

           # # Plotting change in val set ECE for different
           # # plt.figure(figsize=(10,8))
           # # plt.rcParams.update({'font.size': 20})
           # # plt.plot(Ts, eces, 'r-')
           # # plt.xlabel('Temperature values')
           # # plt.ylabel('Val set ECE')
           # # plt.show()

           # # Evaluating the model at T = T_opt
           # # Getting the number of bins
           # net = GlobalPoolingCNN(keep_prob=0.7, temp=T_opt)
           # net.cuda()
           # net.load_state_dict(torch.load(args.save_loc + saved_model_name))

           # conf_matrix, accuracy, labels, predictions, confidences = test_classification_net(embedding_model, net, x_val, y_val, device)
           # ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
           # mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
           # nll, _ = test_cross_entropy(1, embedding_model, net, x_val, y_val, device, 'Test')
            scaled_model = ModelWithTemperature_ng(embedding_model, net, log=args.log)
            scaled_model.set_temperature_cv(x_pval, y_pval)
            T_opt = scaled_model.get_temperature()
            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net_ts(scaled_model, x_val, y_val, device)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            l2 = l2_error(confidences, predictions, labels, num_bins=num_bins)
            nll = test_cross_entropy_ts(1, scaled_model, x_val, y_val, device, 'Test')
            # print ('Optimal temperature by training on nll: ' + str(T_opt))
        
        elif args.cross_validate_on_adaece:
            scaled_model = ModelWithTemperature_ng(embedding_model, net, log=args.log)
            scaled_model, p_ece, ece = scaled_model.set_temperature_cv(x_pval, y_pval, use_adaptive_ECE=True, x_test=x_val, y_test=y_val)
            T_opt = scaled_model.get_temperature()
            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net_ts(scaled_model, x_val, y_val, device)
            #ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            nll = test_cross_entropy_ts(1, scaled_model, x_val, y_val, device, 'Test')

        elif args.cross_validate_on_sce:
            scaled_model = ModelWithTemperature_ng(embedding_model, net, log=args.log)
            scaled_model, p_ece, ece = scaled_model.set_temperature_cv(x_pval, y_pval, use_SCE=True, x_test=x_val, y_test=y_val)
            T_opt = scaled_model.get_temperature()
            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net_ts(scaled_model, x_val, y_val, device)
            #ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            nll = test_cross_entropy_ts(1, scaled_model, x_val, y_val, device, 'Test')

        elif args.train_on_nll:
            scaled_model = ModelWithTemperature_ng(embedding_model, net, log=args.log)
            scaled_model.set_temperature(x_pval, y_pval)
            T_opt = scaled_model.get_temperature()
            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net_ts(scaled_model, x_val, y_val, device)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            nll = test_cross_entropy_ts(1, scaled_model, x_val, y_val, device, 'Test')
            # print ('Optimal temperature by training on nll: ' + str(T_opt))
        
        elif args.train_all:
            scaled_model = ModelWithTemperature_ng(embedding_model, net, log=args.log)
            scaled_model, p_ece, ece, p_aec, aec, p_sce, sce = scaled_model.set_temperature_cv(x_pval, y_pval, use_ALL=True, x_test=x_val, y_test=y_val)
            T_opt = scaled_model.get_temperature()
            conf_matrix, accuracy, labels, predictions, confidences = test_classification_net_ts(scaled_model, x_val, y_val, device)
            #ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            nll = test_cross_entropy_ts(1, scaled_model, x_val, y_val, device, 'Test')
        #print (conf_matrix)
        # print ('Test error: ' + str((1 - accuracy)))
        # print ('Test NLL: ' + str(nll))
        # print ('ECE: ' + str(ece))
        # print ('MCE: ' + str(mce))
        
        if args.train_all:    
            print('& {:s} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}({:.2f}) & {:.4f} & {:.4f} & {:.4f} & {:.4f}\\\\'.format(saved_model_name, 1-p_accuracy, p_nll, p_ece, p_aec, p_sce, p_mce, nll, T_opt,  ece,  aec, sce, mce))
        else:
            # Test NLL & ECE & MCE
            #print('& {:s} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}({:.2f}) & {:.4f} & {:.4f}\\\\'.format(saved_model_name,  1-p_accuracy,  p_nll,  p_ece,  p_mce,  nll,  T_opt,  ece,  mce))
            print('& {:s} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}({:.2f}) & {:.4f} & {:.4f} & {:.4f}\\\\'.format(saved_model_name, 1-p_accuracy, p_nll, p_ece, p_mce, p_l2, nll, T_opt, ece, mce, l2))

        # Plotting the reliability plot
        # reliability_plot(confidences, predictions, labels, num_bins=num_bins)
        # bin_strength_plot(confidences, predictions, labels, num_bins=num_bins)

