from tqdm import tqdm
import torch
from torch.autograd import Variable as Var
from utils import map_label_to_target, map_label_to_target_sentiment
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import gc

import pdb

class SentimentTrainer(object):
    """
    For Sentiment module
    """
    def __init__(self, args, model, embedding_model ,criterion, optimizer, loss, temp = 1.0, train_and_test = True):
        super(SentimentTrainer, self).__init__()
        self.args       = args
        self.model      = model
        self.embedding_model = embedding_model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.loss       = loss
        self.input      = []
        self.target     = []
        self.train_and_test       = train_and_test
        self.temp       = temp

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        # torch.manual_seed(789)
        if self.train_and_test and self.loss == 'mmce_weighted':
            self.criterion.initialize_list()
            gc.collect()
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            tree, sent, label = dataset[indices[idx]]
            input = Var(sent)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain))
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output, err = self.model.forward(tree, emb, training = True) #What does the output signify here? Output is the predicted log probab for each class for the entire tree
            #params = self.model.childsumtreelstm.getParameters()
            # params_norm = params.norm()
            err = err/self.args.batchsize # + 0.5*self.args.reg*params_norm*params_norm # custom bias
            loss += err.data.item() #
            if self.train_and_test and self.loss == 'mmce_weighted':
                err.backward(retain_graph=True)
            else:
                err.backward()
            k += 1
            if k==self.args.batchsize:
                if self.train_and_test and self.loss == 'mmce_weighted':
                    err = self.criterion.get_data_len()*self.criterion.forward_mmce()/self.args.batchsize
                    self.criterion.initialize_list()
                    gc.collect()
                    loss += err.data.item()
                    err.backward()
                for f in self.embedding_model.parameters():
                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        loss, k = 0, 0
        predictions = torch.zeros(len(dataset))
        if self.train_and_test and self.loss == 'mmce_weighted':
            self.criterion.initialize_list()
            gc.collect()
        #predictions = predictions
        indices = torch.range(1,dataset.num_classes)
        # for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(dataset)):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            output, _ = self.model(tree, emb) # size(1,5)
            output = output / self.temp
            err = self.criterion(output, target)
            if self.train_and_test and self.loss == 'mmce_weighted':
                self.input.append(output)
                self.target.append(target)
            loss += err.data.item()
            k += 1
            if k==self.args.batchsize:
                if self.train_and_test and self.loss == 'mmce_weighted':
                    self.criterion.set_list(self.input, self.target)
                    err = self.criterion.get_data_len()*self.criterion.forward_mmce()
                    self.criterion.initialize_list()
                    gc.collect()
                    loss += err.data.item()
                k = 0

            output[:,1] = -9999 # no need middle (neutral) value
            val, pred = torch.max(output, 1)
            #predictions[idx] = pred.data.cpu()[0][0]
            predictions[idx] = pred.data.cpu()[0]
            # predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions

    def test_classification_net(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        loss = 0
        labels_list = []
        predictions_list = []
        confidence_vals_list = []
        indices = torch.range(1,dataset.num_classes)
        # for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(dataset)):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            output, _ = self.model(tree, emb) # size(1,5)
            output = output / self.temp
            err = self.criterion(output, target)
            loss += err.data.item()
            softmax = F.softmax(output, dim=1)
            softmax[:,1] = -9999 # no need middle (neutral) value
            confidence_val, prediction = torch.max(softmax, dim=1)

            labels_list.append(target.data.cpu()[0])
            predictions_list.append(prediction.data.cpu()[0])
            confidence_vals_list.append(confidence_val.data.cpu()[0])

        accuracy = accuracy_score(labels_list, predictions_list)
        return loss/len(dataset), confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
            predictions_list, confidence_vals_list

    def test_classification_net_sst(self, scaled_model, dataset):
        scaled_model.eval()
        loss = 0
        labels_list = []
        predictions_list = []
        confidence_vals_list = []
        indices = torch.range(1,dataset.num_classes)
        # for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(dataset)):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = Var(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=self.args.fine_grain), volatile=True)
            if self.args.cuda:
                input = input.cuda()
                target = target.cuda()
            output = scaled_model(input, tree)
            err = self.criterion(output, target)
            loss += err.data.item()
            softmax = F.softmax(output, dim=1)
            softmax[:,1] = -9999 # no need middle (neutral) value
            confidence_val, prediction = torch.max(softmax, dim=1)

            labels_list.append(target.data.cpu()[0])
            predictions_list.append(prediction.data.cpu()[0])
            confidence_vals_list.append(confidence_val.data.cpu()[0])

        accuracy = accuracy_score(labels_list, predictions_list)
        return loss/len(dataset), confusion_matrix(labels_list, predictions_list), accuracy, labels_list,\
            predictions_list, confidence_vals_list

    def set_temp(self, temp):
        self.temp = temp


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch '+str(self.epoch+1)+''):
            ltree,lsent,rtree,rsent,label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            target = Var(map_label_to_target(label,dataset.num_classes))
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)
            loss += err.data.item()
            err.backward()
            k += 1
            if k%self.args.batchsize==0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.epoch += 1
        return loss/len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.range(1,dataset.num_classes)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            ltree,lsent,rtree,rsent,label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            target = Var(map_label_to_target(label,dataset.num_classes), volatile=True)
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree,linput,rtree,rinput)
            err = self.criterion(output, target)
            loss += err.data.item()
            predictions[idx] = torch.dot(indices,torch.exp(output.data.cpu()))
        return loss/len(dataset), predictions
