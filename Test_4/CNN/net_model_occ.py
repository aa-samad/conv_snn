from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
num_classes = 10
batch_size = 100
learning_rate = 1e-3
num_epochs = 20  # max epoch

# cnn_layer(in_planes(channels), out_planes(channels), stride, padding, kernel_size)
cfg_cnn = [(1, 256, 1, 1, 3),
           (256, 256, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]         # conv layers input image shape (+ last output shape)
# fc layer
cfg_fc = [128, 10]              # linear layers output


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(10, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchn1 = nn.BatchNorm2d(out_planes)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchn2 = nn.BatchNorm2d(out_planes)
        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window=10):
        x = input.cuda()

        x = F.relu(self.conv1(x.float()))
        x = self.batchn1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = self.batchn2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(batch_size * 2, -1)      # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        outputs = F.softmax(x, dim=1)
        return outputs


