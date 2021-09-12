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


from Losses.focal_loss import FocalLoss
from Losses.focal_loss_adaptive_gamma_sst import FocalLossAdaptive
from Losses.mmce_sst import MMCE_weighted
from Losses.brier_score import BrierScore
from Losses.cross_entropy_smoothed import CrossEntropySmoothed

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
    print(args)
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
    print('==> SST vocabulary size : %d ' % vocab.size())

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

    if args.loss_function == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        args.name = 'cross_entropy'

    elif args.loss_function == 'cross_entropy_smoothed':
        criterion = CrossEntropySmoothed(smoothing=args.smoothing, num_classes=args.num_classes)
        args.name = 'cross_entropy_smoothed_' + str(args.smoothing)

    elif args.loss_function == 'focal_loss':
        if args.gamma_schedule == 1:
            criterion = FocalLoss(gamma=args.gamma)
            args.name = 'focal_loss_scheduled_gamma_' + str(args.gamma) + '_' + str(args.gamma2) + '_' + str(args.gamma3)
        else:
            criterion = FocalLoss(gamma=args.gamma)
            args.name = 'focal_loss_gamma_' + str(args.gamma)

    elif args.loss_function == 'focal_loss_adaptive':
        criterion = FocalLossAdaptive(gamma=args.gamma)
        args.name = 'focal_loss_adaptive_gamma_' + str(args.gamma)

    elif args.loss_function == 'mmce_weighted':
        #criterion = mmce_weighted(lamda=args.lamda)
        criterion = MMCE_weighted()
        args.name = 'mmce_weighted_lamda_' + str(args.lamda)
    
    elif args.loss_function == 'brier_score':
        criterion = BrierScore()
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

    utils.count_param(model)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sst_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:

        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove,'glove.840B.300d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(),glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        print('done creating emb, quit')

    if is_preprocessing_data:
        print ('done preprocessing data, quit program to prevent memory leak')
        print ('please run again')
        quit()

    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()

    # model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)
    embedding_model.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer     = SentimentTrainer(args, model, embedding_model ,criterion, optimizer, args.loss_function)

    mode = 'EXPERIMENT'
    if mode == 'DEBUG':
        for epoch in range(args.epochs):
            dev_loss = trainer.train(dev_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            test_loss, test_pred = trainer.test(test_dataset)

            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Dev loss   : %f \t' % dev_loss, end="")
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
    elif mode == "PRINT_TREE":
        for i in range(0, 10):
            ttree, tsent, tlabel = dev_dataset[i]
            utils.print_tree(ttree, 0)
            print('_______________')
        print('break')
        quit()
    elif mode == "EXPERIMENT":
        max_dev = 0
        max_dev_epoch = 0
        filename = args.name + '.pth'
        for epoch in range(args.epochs):
            if args.gamma_schedule == 1 and (epoch >= 3 and epoch < 7):
                criterion.set_gamma(args.gamma2)
            elif args.gamma_schedule == 1 and epoch >= 7:
                criterion.set_gamma(args.gamma3)
            train_loss = trainer.train(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            torch.save(model, args.saved + str(epoch) + '_model_' + filename)
            torch.save(embedding_model, args.saved + str(epoch) + '_embedding_' + filename)
            if dev_acc > max_dev:
                max_dev = dev_acc
                max_dev_epoch = epoch
            gc.collect()
        print('epoch ' + str(max_dev_epoch) + ' dev score of ' + str(max_dev))
        print('eva on test set ')
        model = torch.load(args.saved + str(max_dev_epoch) + '_model_' + filename)
        embedding_model = torch.load(args.saved + str(max_dev_epoch) + '_embedding_' + filename)
        trainer = SentimentTrainer(args, model, embedding_model, criterion, optimizer, args.loss_function)
        test_loss, test_pred = trainer.test(test_dataset)
        test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
        print('Epoch with max dev:' + str(max_dev_epoch) + ' |test percentage ' + str(test_acc))
        print('____________________' + str(args.name) + '___________________')
    else:
        for epoch in range(args.epochs):
            train_loss = trainer.train(train_dataset)
            train_loss, train_pred = trainer.test(train_dataset)
            dev_loss, dev_pred = trainer.test(dev_dataset)
            test_loss, test_pred = trainer.test(test_dataset)

            train_acc = metrics.sentiment_accuracy_score(train_pred, train_dataset.labels)
            dev_acc = metrics.sentiment_accuracy_score(dev_pred, dev_dataset.labels)
            test_acc = metrics.sentiment_accuracy_score(test_pred, test_dataset.labels)
            print('==> Train loss   : %f \t' % train_loss, end="")
            print('Epoch ', epoch, 'train percentage ', train_acc)
            print('Epoch ', epoch, 'dev percentage ', dev_acc)
            print('Epoch ', epoch, 'test percentage ', test_acc)


if __name__ == "__main__":
    # log to console and file
    logger1 = log_util.create_logger("temp_file", print_console=True)
    logger1.info("LOG_FILE") # log using loggerba
    # attach log to stdout (print function)
    s1 = log_util.StreamToLogger(logger1)
    sys.stdout = s1
    print ('_________________________________start___________________________________')
    main()
