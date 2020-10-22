'''
Implementation of the MMCE (MMCE_m) and MMCE_weighted (MMCE_w).
Reference:
[1]  A. Kumar, S. Sarawagi, U. Jain, Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings.
     ICML, 2018.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MMCE(nn.Module):
    """
    Computes MMCE_m loss.
    """
    def __init__(self, device):
        super(MMCE, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1) #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(torch.eq(pred_labels, target),
                          torch.ones(pred_labels.shape).to(self.device),
                          torch.zeros(pred_labels.shape).to(self.device))

        c_minus_r = correct_mask - predicted_probs

        dot_product = torch.mm(c_minus_r.unsqueeze(1),
                        c_minus_r.unsqueeze(0))

        prob_tiled = predicted_probs.unsqueeze(1).repeat(1, predicted_probs.shape[0]).unsqueeze(2)
        prob_pairs = torch.cat([prob_tiled, prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        kernel_prob_pairs = self.torch_kernel(prob_pairs)

        numerator = dot_product*kernel_prob_pairs
        #return torch.sum(numerator)/correct_mask.shape[0]**2
        return torch.sum(numerator)/torch.pow(torch.tensor(correct_mask.shape[0]).type(torch.FloatTensor),2)



class MMCE_weighted(nn.Module):
    """
    Computes MMCE_w loss.
    """
    def __init__(self, device):
        super(MMCE_weighted, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

    def get_pairs(self, tensor1, tensor2):
        correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
        incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

        correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
                                    dim=2)
        incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
                                    dim=2)

        correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
        incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

        correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
                                    dim=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    def get_out_tensor(self, tensor1, tensor2):
        return torch.mean(tensor1*tensor2)

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = F.softmax(input, dim=1)
        predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

        correct_mask = torch.where(torch.eq(predicted_labels, target),
                                    torch.ones(predicted_labels.shape).to(self.device),
                                    torch.zeros(predicted_labels.shape).to(self.device))

        k = torch.sum(correct_mask).type(torch.int64)
        k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
        cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
        k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
        k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (correct_mask.shape[0] - 2))


        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

        correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

        correct_kernel = self.torch_kernel(correct_prob_pairs)
        incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)  

        sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

        correct_correct_vals = self.get_out_tensor(correct_kernel,
                                                          sampling_weights_correct)
        sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

        incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
                                                          sampling_weights_incorrect)
        sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

        correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
                                                          sampling_correct_incorrect)

        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)

        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)
        return torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))