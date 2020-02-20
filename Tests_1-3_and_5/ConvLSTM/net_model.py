from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling3D
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import numpy as np
import pylab as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
num_classes = 10
batch_size = 100
learning_rate = 1e-3
num_epochs = 20  # max epoch


seq = Sequential()
seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                   input_shape=(10, 28, 28, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid'))

seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(MaxPooling3D(pool_size=(1, 2, 2), padding='valid'))

seq.add(Flatten())
seq.add(Dense(128, activation='relu'))
seq.add(Dropout(0.5))
seq.add(Dense(num_classes * 2, activation='softmax'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
seq.compile(loss='binary_crossentropy', optimizer=optimizer)



