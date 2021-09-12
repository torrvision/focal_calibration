from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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
from utils import map_label_to_target, map_label_to_target_sentiment
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import SentimentTrainer

from Experiments.error_bars_sst import get_calibration_error_uncertainties
#from Experiments.error_bars_sst_smoothing import get_calibration_error_uncertainties #TODO: to be used for CE smoothing trained model
from Metrics.metrics import expected_calibration_error
from Metrics.metrics import maximum_calibration_error
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
    opt_temp = args.temp
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
        model.eval()
        embedding_model.eval()
        logits_list = []
        labels_list = []
        batch_size = 128
        #for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(test_dataset)):
            tree, sent, label = test_dataset[idx]
            input = Variable(sent, volatile=True).cuda()
            target = Variable(map_label_to_target_sentiment(label,test_dataset.num_classes, fine_grain=0), volatile=True).cuda()
            emb = F.torch.unsqueeze(embedding_model(input),1)
            logits_var, _ = model(tree, emb)
            logits_list.append(logits_var.data)
            labels_list.append(target.data)

        all_logits = Variable(torch.cat(logits_list).cuda())
        all_labels = Variable(torch.cat(labels_list).cuda())
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
        print(filename)
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
        
if __name__ == "__main__":
    # log to console and file
    logger1 = log_util.create_logger("temp_file", print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    # print ('_________________________________start___________________________________')
    main()
