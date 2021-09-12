from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
import gc
import sys
from meowlogtool import log_util

sys.path.append("treeLSTM")
import utils
# IMPORT CONSTANTS
import Constants
# NEURAL NETWORK MODULES/LAYERS
from model import *
# DATA HANDLING CLASSES
from tree import Tree
from vocab import Vocab
# DATASET CLASS FOR SICK DATASET
from dataset import SSTDataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# UTILITY FUNCTIONS
from utils import load_word_vectors, build_vocab
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer

from Experiments.temperature_scaling_sst import ModelWithTemperature_sst
#from Experiments.temperature_scaling_sst_smoothing import ModelWithTemperature_sst #TODO: to be used for CE smoothing trained model
from Metrics.metrics import expected_calibration_error
from Metrics.metrics import maximum_calibration_error
from Metrics.metrics import l2_error
from Metrics.plots import reliability_plot, bin_strength_plot

from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma_sst import FocalLossAdaptive
from Losses.mmce_sst import MMCE_weighted
from Losses.cross_entropy_smoothed import CrossEntropySmoothed

import pdb

# MAIN BLOCK
def main():
    global args
    args = parse_args(type=1)
    args.input_dim= 300
    if args.model_name == 'dependency':
        args.mem_dim = 168
    elif args.model_name == 'constituency':
        args.mem_dim = 150
    if args.fine_grain:
        args.num_classes = 5 # 0 1 2 3 4
    else:
        args.num_classes = 3 # 0 1 2 (1 neutral)
    args.cuda = args.cuda and torch.cuda.is_available()
    # args.cuda = False
    # print(args)
    # torch.manual_seed(args.seed)
    # if args.cuda:
        # torch.cuda.manual_seed(args.seed)

    train_dir = os.path.join(args.data,'train/')
    dev_dir = os.path.join(args.data,'dev/')
    test_dir = os.path.join(args.data,'test/')

    # write unique words from all token files
    token_files = [os.path.join(split, 'sents.toks') for split in [train_dir, dev_dir, test_dir]]
    vocab_file = os.path.join(args.data,'vocab-cased.txt') # use vocab-cased
    # build_vocab(token_files, vocab_file) NO, DO NOT BUILD VOCAB,  USE OLD VOCAB

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file)
    # print('==> SST vocabulary size : %d ' % vocab.size())

    # Load SST dataset splits

    is_preprocessing_data = False # let program turn off after preprocess data

    # train
    train_file = os.path.join(args.data,'sst_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SSTDataset(train_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(train_dataset, train_file)
        is_preprocessing_data = True

    # dev
    dev_file = os.path.join(args.data,'sst_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SSTDataset(dev_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(dev_dataset, dev_file)
        is_preprocessing_data = True

    # test
    test_file = os.path.join(args.data,'sst_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes, args.fine_grain, args.model_name)
        torch.save(test_dataset, test_file)
        is_preprocessing_data = True

    #criterion = nn.NLLLoss()

    criterion = nn.CrossEntropyLoss()
    if args.loss_function == 'cross_entropy':        
        args.name = 'cross_entropy'

    elif args.loss_function == 'cross_entropy_smoothed':
        args.name = 'cross_entropy_smoothed_' + str(args.smoothing)

    elif args.loss_function == 'focal_loss':
        if args.gamma_schedule == 1:
            args.name = 'focal_loss_scheduled_gamma_' + str(args.gamma) + '_' + str(args.gamma2) + '_' + str(args.gamma3)
        else:
            args.name = 'focal_loss_gamma_' + str(args.gamma)

    elif args.loss_function == 'focal_loss_adaptive':
        args.name = 'focal_loss_adaptive_gamma_' + str(args.gamma)

    elif args.loss_function == 'mmce_weighted':
        args.name = 'mmce_weighted_lamda_' + str(args.lamda)
    
    elif args.loss_function == 'brier_score':
        args.name = 'brier_score'

    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTMSentiment(
                args.cuda, vocab.size(),
                args.input_dim, args.mem_dim,
                args.num_classes, args.model_name, criterion
            )

    embedding_model = nn.Embedding(vocab.size(), args.input_dim)

    if args.cuda:
        embedding_model = embedding_model.cuda()

    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim=='adam':
        optimizer   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        # optimizer   = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
        optimizer = optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:

        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        # print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(),glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        # print('done creating emb, quit')

    if is_preprocessing_data:
        # print ('done preprocessing data, quit program to prevent memory leak')
        # print ('please run again')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    filename = args.name + '.pth'
    model = torch.load(args.saved + str(args.max_dev_epoch) + '_model_' + filename)
    embedding_model = torch.load(args.saved + str(args.max_dev_epoch) + '_embedding_' + filename)

    # create trainer object for training and testing
    trainer     = SentimentTrainer(args, model, embedding_model ,criterion, optimizer, args.loss_function, train_and_test = False)

    num_bins = args.num_bins
    mode = 'EXPERIMENT'
    if mode == "EXPERIMENT":
        p_nll, conf_matrix, p_accuracy, labels, predictions, confidences = trainer.test_classification_net(test_dataset)
        p_ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        p_mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
        p_l2 = l2_error(confidences, predictions, labels, num_bins=num_bins)
        # Printing the required evaluation metrics
        #print (conf_matrix)
        # print ('Test error: ' + str((1 - p_accuracy)))
        # print ('Test NLL: ' + str(p_nll))
        # print ('ECE: ' + str(p_ece))
        # print ('MCE: ' + str(p_mce))

        if args.cross_validate_on_nll:
            # Tuning the temperature parameter on the val set
            T = 0.1
            nll_val = 10 ** 7
            T_opt = 0.1
            Ts = []
            nlls = []
            for i in range(100):
                trainer.set_temp(temp = T)
                nll, _ = trainer.test(dev_dataset)
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

            trainer.set_temp(temp = T_opt)
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net(test_dataset)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)


        elif args.cross_validate_on_ece:
            # Tuning the temperature parameter on the val set
           # T = 0.1
           # ece_val = 10 ** 7
           # T_opt = 0.1
           # Ts = []
           # eces = []
           # for i in range(100):
           #     trainer.set_temp(temp = T)
           #     _, _, _, labels, predictions, confidences = trainer.test_classification_net(dev_dataset)
           #     ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)

           #     # print ('At temperature: ' + str(T) + ' Val set ECE: ' + str(ece))
           #     if (ece < ece_val):
           #         ece_val = ece
           #         T_opt = T
           #     Ts.append(T)
           #     eces.append(ece)
           #     T = T + 0.1
            scaled_model = ModelWithTemperature_sst(embedding_model, model, args.log)
            scaled_model.set_temperature_cv(dev_dataset)
            T_opt = scaled_model.get_temperature()
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net_sst(scaled_model, test_dataset)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            l2 = l2_error(confidences, predictions, labels, num_bins=num_bins)
            
            # print ('Optimal temperature by cross validation on ece: ' + str(T_opt))

            # Plotting change in val set ECE for different
            # plt.figure(figsize=(10,8))
            # plt.rcParams.update({'font.size': 20})
            # plt.plot(Ts, eces, 'r-')
            # plt.xlabel('Temperature values')
            # plt.ylabel('Val set ECE')
            # plt.show()

            # Evaluating the model at T = T_opt
            # Getting the number of bins
        
        elif args.cross_validate_on_adaece:
            scaled_model = ModelWithTemperature_sst(embedding_model, model, args.log)
            scaled_model, p_ece, ece = scaled_model.set_temperature_cv(dev_dataset, use_adaptive_ECE=True, test_dataset=test_dataset)
            T_opt = scaled_model.get_temperature()
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net_sst(scaled_model, test_dataset)
            #ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)

        elif args.cross_validate_on_sce:
            scaled_model = ModelWithTemperature_sst(embedding_model, model, args.log)
            scaled_model, p_ece, ece = scaled_model.set_temperature_cv(dev_dataset, use_SCE=True, test_dataset=test_dataset)
            T_opt = scaled_model.get_temperature()
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net_sst(scaled_model, test_dataset)
            #ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)

        elif args.train_on_nll:
            scaled_model = ModelWithTemperature_sst(embedding_model, model, args.log)
            scaled_model.set_temperature(dev_dataset)
            T_opt = scaled_model.get_temperature()
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net_sst(scaled_model, test_dataset)
            ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            # print ('Optimal temperature by training on nll: ' + str(T_opt))

        elif args.train_all:
            scaled_model = ModelWithTemperature_sst(embedding_model, model, args.log)
            scaled_model, p_ece, ece, p_aec, aec, p_sce, sce = scaled_model.set_temperature_cv(dev_dataset, use_ALL=True, test_dataset=test_dataset)
            T_opt = scaled_model.get_temperature()
            nll, conf_matrix, accuracy, labels, predictions, confidences = trainer.test_classification_net_sst(scaled_model, test_dataset)
            # ece = expected_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            mce = maximum_calibration_error(confidences, predictions, labels, num_bins=num_bins)
            print('& {:s} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f}({:.2f}) & {:.4f} & {:.4f} & {:.4f} & {:.4f}\\\\'.format(str(args.max_dev_epoch) + '_model_' + filename, 1-p_accuracy, p_nll, p_ece, p_aec, p_sce, p_mce, nll, T_opt,  ece,  aec, sce, mce))
            return

        #print (conf_matrix)
        # print ('Test error: ' + str((1 - accuracy)))
        # print ('Test NLL: ' + str(nll))
        # print ('ECE: ' + str(ece))
        # print ('MCE: ' + str(mce))

        # Test NLL & ECE & MCE
        print('& {:s} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} &{:.4f}({:.2f}) & {:.4f} & {:.4f} & {:.4f}\\\\'.format(str(args.max_dev_epoch) + '_model_' + filename,  1-p_accuracy,  p_nll,  p_ece,  p_mce,  p_l2, nll,  t_opt, ece,  mce, l2))
        #print('& {:s} & {:.4f} & {:.4f} & {:.4f}({:.2f}) & {:.4f}\\\\'.format(str(args.max_dev_epoch) + '_model_' + filename,  p_nll,  p_l2,  nll,  T_opt,  l2))
        #print (saved_model_name[:-9] + ' & ' + str(1-p_accuracy) + ' & ' + str(p_nll) + ' & ' + str(p_ece) + ' & ' + str(p_mce) + ' & ' + str(nll) +'(' + str(T_opt) + ')' + ' & ' + str(ece) + ' & ' + str(mce))

        # Plotting the reliability plot
        # reliability_plot(confidences, predictions, labels, num_bins=num_bins)
        # bin_strength_plot(confidences, predictions, labels, num_bins=num_bins)


if __name__ == "__main__":
    # log to console and file
    logger1 = log_util.create_logger("temp_file", print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    # print ('_________________________________start___________________________________')
    main()
