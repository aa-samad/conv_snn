from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled=False
print("device", device)
num_classes = 20
batch_size = 100
learning_rate = 1e-3
num_epochs = 20  # max epoch
LSTM_out_size = 128



class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)   # 28 x 28
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)   # 14 x 14
        self.lstm1 = nn.LSTM(7 * 7 * 128, LSTM_out_size)  # Input dim is 3, output dim is 3
        self.fc1 = nn.Linear(LSTM_out_size, num_classes)
        # self.fc2 = nn.Linear(20, num_classes)


    def forward(self, input, time_window=10):
        # linear output layer layer 
        outputs = torch.zeros(batch_size * 2, num_classes, device=device)
        hidden = torch.randn(2, 1, 1, LSTM_out_size).cuda()

        for step in range(time_window):  # simulation time steps
            x = input[:, step: step + 1, :, :]
            x = F.relu(self.conv1(x.float()))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(batch_size * 2, -1)      # flatten
            x, hidden = self.lstm1(x.view(batch_size * 2, 1, -1), hidden)
            x = x.view(batch_size * 2, -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        outputs = F.softmax(x, dim=1)
        return outputs


