import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable
from utils import map_label_to_target, map_label_to_target_sentiment

import pdb

class ModelWithTemperature_sst(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, embedding_model, model, log = True):
        super(ModelWithTemperature_sst, self).__init__()
        self.embedding_model = embedding_model
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.log = log

    def forward(self, input, tree):
        self.model.eval()
        self.embedding_model.eval()
        emb = F.torch.unsqueeze(self.embedding_model(input),1)
        output, _ = self.model(tree, emb)
        return self.temperature_scale(output)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, dataset):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        self.model.eval()
        self.embedding_model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss_sst().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        batch_size = 128
        #for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(dataset)):
            tree, sent, label = dataset[idx]
            input = Variable(sent, volatile=True).cuda()
            target = Variable(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=0), volatile=True).cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            logits_var, _ = self.model(tree, emb)
            logits_list.append(logits_var.data)
            labels_list.append(target.data)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        logits = Variable(logits)
        labels = Variable(labels)

        before_temperature_nll = nll_criterion(logits, labels).data[0]
        before_temperature_ece = ece_criterion(logits, labels).data[0]
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        
        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        if 0:
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]
            if self.log:
                print('Optimal temperature: %.3f' % self.temperature.data[0])
                print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        else:
            best_ece_val = 10 ** 7                                                                                                                                                                          
            best_T = 1
            best_hyper_param = ''
            lr_list = [0.001, 0.003, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5]
            max_iter_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500]
            #lr_list = [0.001, 0.003, 0.005, 0.007, 0.01, 0.013, 0.015, 0.017, 0.02, 0.023, 0.025, 0.027, 0.03, 0.033, 0.035, 0.037, 0.04, 0.043, 0.045, 0.047, 0.05, 0.053, 0.055, 0.057, 0.06, 0.063, 0.065, 0.067, 0.07, 0.073, 0.075, 0.077, 0.08, 0.083, 0.085, 0.087, 0.09, 0.093, 0.095, 0.097, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            #max_iter_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1500]
            for lr in lr_list:
                for max_iter in max_iter_list:
                    self.temperature = nn.Parameter(torch.ones(1) * 1.5)
                    self.cuda()
                    optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
                    optimizer.zero_grad()
                    optimizer.step(eval)

                    # Calculate NLL and ECE after temperature scaling
                    after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
                    after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]

                    if best_ece_val > after_temperature_ece and self.temperature.data[0] < 100 and self.temperature.data[0] > 0:
                        best_ece_val = after_temperature_ece
                        best_T = self.temperature.data[0]
                        best_hyper_param = 'LBFGS ' + str(lr) + ' ' + str(max_iter)

                    if self.log:
                        print('Optimizing using LBFGS')
                        print('lr: '+str(lr)+', max_iter: '+str(max_iter))
                        print('Optimal temperature: %.3f' % self.temperature.data[0])
                        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
                        print(best_hyper_param)

            for lr in lr_list:
                for max_iter in max_iter_list:
                    self.temperature = nn.Parameter(torch.ones(1) * 1.5)
                    self.cuda()
                    optimizer = optim.SGD([self.temperature], lr=lr, momentum=0.9)
                    
                    for i in range(max_iter):
                        optimizer.zero_grad()
                        optimizer.step(eval)
                    
                    # Calculate NLL and ECE after temperature scaling
                    after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
                    after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]

                    if best_ece_val > after_temperature_ece and self.temperature.data[0] < 100 and self.temperature.data[0] > 0:
                        best_ece_val = after_temperature_ece
                        best_T = self.temperature.data[0]
                        best_hyper_param = 'SGD ' + str(lr) + ' ' + str(max_iter)

                    if self.log:
                        print('Optimizing using SGD')
                        print('lr: '+str(lr)+', max_iter: '+str(max_iter))
                        print('Optimal temperature: %.3f' % self.temperature.data[0])
                        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
                        print(best_hyper_param)
           
            for lr in lr_list:
                for max_iter in max_iter_list:
                    self.temperature = nn.Parameter(torch.ones(1) * 1.5)
                    self.cuda()
                    optimizer = optim.Adam([self.temperature], lr=lr)
                    
                    for i in range(max_iter):
                        optimizer.zero_grad()
                        optimizer.step(eval)
                    
                    # Calculate NLL and ECE after temperature scaling
                    after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
                    after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]

                    if best_ece_val > after_temperature_ece and self.temperature.data[0] < 100 and self.temperature.data[0] > 0:
                        best_ece_val = after_temperature_ece
                        best_T = self.temperature.data[0]
                        best_hyper_param = 'Adam ' + str(lr) + ' ' + str(max_iter)

                    if self.log:
                        print('Optimizing using Adam')
                        print('lr: '+str(lr)+', max_iter: '+str(max_iter))
                        print('Optimal temperature: %.3f' % self.temperature.data[0])
                        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
                        print(best_hyper_param)


            self.temperature = nn.Parameter(torch.ones(1) * best_T)
            self.cuda()

        # Hyper-param tuning, didn't actually work
        # min_nll = 10 ** 7
        # lr_opt = 0.01
        # max_iter_opt = 50
        # T_init_opt = 1.5
        # for T_init in [0.5, 1.0, 1.5]:
        #     for lr in [0.001, 0.01, 0.0001, 0.1]:
        #         for max_iter in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 500]:
        #             self.temperature = nn.Parameter(torch.ones(1) * T_init)
        #             self.cuda()
        #             optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        #             optimizer.step(eval)

        #             # Calculate NLL and ECE after temperature scaling
        #             after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
        #             after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]

        #             if after_temperature_nll < min_nll:
        #                 min_nll = after_temperature_nll
        #                 T_init_opt = T_init
        #                 lr_opt = lr
        #                 max_iter_opt = max_iter
        #             #print('lr:%.4f, max_iter:%.3f, Optimal temperature: %.3f' % (lr, max_iter, self.temperature.data[0]))
        #             #print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        #             #print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        # self.temperature = nn.Parameter(torch.ones(1) * T_init_opt)
        # self.cuda()
        # optimizer = optim.LBFGS([self.temperature], lr=lr_opt, max_iter=max_iter_opt)
        # optimizer.step(eval)
        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]
        # print('lr:%.4f, max_iter:%.3f, Optimal temperature: %.3f' % (lr, max_iter, self.temperature.data[0]))
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self
    
    def set_temperature_cv(self, dataset, use_adaptive_ECE=False, use_SCE=False, use_ALL=False, test_dataset=None):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        self.model.eval()
        self.embedding_model.eval()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if use_ALL:
            ece_criterion = _ECELoss_sst().cuda()
            ec_criterion = _ECELoss_sst().cuda()
            sce_criterion = _SCELoss_sst().cuda()
            aec_criterion = _AdaptiveECELoss().cuda()
        if use_adaptive_ECE:
            ece_criterion = _AdaptiveECELoss().cuda()
        elif use_SCE:
            ece_criterion = _SCELoss_sst().cuda()
        else:
            ece_criterion = _ECELoss_sst().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        batch_size = 128
        #for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
        for idx in range(len(dataset)):
            tree, sent, label = dataset[idx]
            input = Variable(sent, volatile=True).cuda()
            target = Variable(map_label_to_target_sentiment(label,dataset.num_classes, fine_grain=0), volatile=True).cuda()
            emb = F.torch.unsqueeze(self.embedding_model(input),1)
            logits_var, _ = self.model(tree, emb)
            logits_list.append(logits_var.data)
            labels_list.append(target.data)

        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()
        logits = Variable(logits)
        labels = Variable(labels)

        before_temperature_nll = nll_criterion(logits, labels).data[0]
        before_temperature_ece = ece_criterion(logits, labels).data[0]
        if self.log:
            print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        
        #def eval():
        #    loss = nll_criterion(self.temperature_scale(logits), labels)
        #    loss.backward()
        #    return loss

        nll_val = 10 ** 7
        ece_val = 10 ** 7
        T_opt_nll = 1
        T_opt_ece = 1
        T = 0.1 #1
        for i in range(100): #range(90)
        #import numpy as np
        #for T in np.linspace(0.8,1.0,15000):
            self.temperature = nn.Parameter(torch.ones(1) * T)
            self.cuda()
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]
            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            if ece_val > after_temperature_ece:
                T_opt_ece = T
                ece_val = after_temperature_ece
            
            T += 0.1

        self.temperature = nn.Parameter(torch.ones(1) * T_opt_ece) #T_opt_nll
        self.cuda()
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).data[0]
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]

        if test_dataset is not None:
            logits_list = []
            labels_list = []
            #for idx in tqdm(range(len(dataset)),desc='Testing epoch  '+str(self.epoch)+''):
            for idx in range(len(test_dataset)):
                tree, sent, label = test_dataset[idx]
                input = Variable(sent, volatile=True).cuda()
                target = Variable(map_label_to_target_sentiment(label,test_dataset.num_classes, fine_grain=0), volatile=True).cuda()
                emb = F.torch.unsqueeze(self.embedding_model(input),1)
                logits_var, _ = self.model(tree, emb)
                logits_list.append(logits_var.data)
                labels_list.append(target.data)

            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            logits = Variable(logits)
            labels = Variable(labels)

            before_temperature_ece = ece_criterion(logits, labels).data[0]
            after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).data[0]
            if use_ALL:
                before_temperature_ec = ec_criterion(logits, labels).data[0]
                after_temperature_ec = ec_criterion(self.temperature_scale(logits), labels).data[0]
                before_temperature_sce = sce_criterion(logits, labels).data[0]
                before_temperature_aec = aec_criterion(logits, labels).data[0]
                after_temperature_sce = sce_criterion(self.temperature_scale(logits), labels).data[0]
                after_temperature_aec = aec_criterion(self.temperature_scale(logits), labels).data[0]
                return self, before_temperature_ec, after_temperature_ec, before_temperature_aec, after_temperature_aec, before_temperature_sce, after_temperature_sce

            #print('Before Adapt ECE: %.3f, after Adapt ECE: %.3f' % (before_temperature_ece, after_temperature_ece))
            return self, before_temperature_ece, after_temperature_ece

        return self


    def get_temperature(self):
        return self.temperature.data[0]


class _ECELoss_sst(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss_sst, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = Variable(torch.zeros(1)).type_as(logits)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.data[0] > 0:
                accuracy_in_bin = accuracies[Variable(in_bin.data)].float().mean()
                avg_confidence_in_bin = confidences[Variable(in_bin.data)].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class _AdaptiveECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_AdaptiveECELoss, self).__init__()
        self.nbins = n_bins
    
    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #import pdb; pdb.set_trace()
        n, bin_boundaries = np.histogram(confidences.cpu().data.numpy(), self.histedges_equalN(confidences.cpu().data.numpy()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = Variable(torch.zeros(1)).type_as(logits)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.data[0] > 0:
                accuracy_in_bin = accuracies[Variable(in_bin.data)].float().mean()
                avg_confidence_in_bin = confidences[Variable(in_bin.data)].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece


class _SCELoss_sst(nn.Module):
    """
    Calculates the Static Calibration Error of a model.
    See: Nixon, J.; Dusenberry, M.; Zhang, L.; Jerfel, G.; and Tran, D. 2019.
    Measuring calibration in deep learning. arXiv preprint arXiv:1904.01685
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_SCELoss_sst, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        num_classes = int((torch.max(labels) + 1).data[0])
        softmaxes = F.softmax(logits, dim=1)
        softmaxes[:,1] = -9999 # no need middle (neutral) value
        per_class_sce = None

        for i in range(num_classes):
            class_confidences = softmaxes[:, i]
            class_sce = Variable(torch.zeros(1)).type_as(logits)
            labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                in_bin = (class_confidences.gt(bin_lower) * class_confidences.le(bin_upper)).detach()
                prop_in_bin = in_bin.float().mean()
                if prop_in_bin.data[0] > 0:
                    try:
                        accuracy_in_bin = labels_in_class[in_bin].float().mean()
                        avg_confidence_in_bin = class_confidences[in_bin].mean()
                    except:
                        import pdb; pdb.set_trace()
                    class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            if (i == 0):
                per_class_sce = class_sce
            else:
                per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

        sce = torch.mean(per_class_sce)
        return sce