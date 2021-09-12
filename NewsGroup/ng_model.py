import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalPoolingCNN(nn.Module):
    def __init__(self, keep_prob, temp=1.0):
        super(GlobalPoolingCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0)
        self.fc1   = nn.Linear(128, 128)
        self.fc2   = nn.Linear(128, 20)
        self.dropout = nn.Dropout(p=1-keep_prob)
        self.temp = temp

    def forward(self, x): #x: batchsize x step x 100
        out = F.relu(self.conv1(x.transpose(1,2)))
        out = F.max_pool1d(out, kernel_size=5, stride=1)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, kernel_size=5, stride=1)
        out = F.relu(self.conv3(out)) 
        # batch x step x feature_size
        # now global max pooling layer
        out, _ = torch.max(out, dim=2) #The number of channels are preserved to be 128 and max is along step
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out) / self.temp
        return out
