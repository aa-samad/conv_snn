import struct
import numpy as np
import scipy.misc
import h5py
import glob
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat
import sys
import cv2


train_filename = "DvsGesture/dvs_gestures_dense_train.pt"
test_filename = "DvsGesture/dvs_gestures_dense_test.pt"

mapping = { 0 :'airplane'  ,
            1 :'automobile',
            2 :'bird' ,
            3 :'cat'   ,
            4 :'deer'  ,
            5 :'dog'    ,
            6 :'frog'   ,
            7 :'horse'       ,
            8 :'ship'      ,
            9 :'truck'     }


class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '{}.pt'.format(index))

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


def gather_addr(directory, start_id, end_id):
    fns = []
    for i in range(start_id, end_id):
        for class0 in mapping.keys():
            file_name = directory + '/' + mapping[class0] + '/' + "{}".format(i) + '.mat'
            fns.append(file_name)
    return fns


def events_to_frames(filename, dt):
    label_filename = filename[:].split('/')[1]
    label = int(list(mapping.values()).index(label_filename))
    frames = np.zeros((10, 2, 128, 128))
    events = loadmat(filename)['out1']
    # --- normal event concatanation
    # for i in range(10):
    #     frames[i, events[i * dt: (i+1) * dt, 3], events[i * dt: (i+1) * dt, 1], events[i * dt: (i+1) * dt, 2]] += 1
    # --- building time surfaces
    for i in range(10):
        r1 = i * (events.shape[0] // 10)
        r2 = (i + 1) * (events.shape[0] // 10)
        frames[i, events[r1:r2, 3], events[r1:r2, 1], events[r1:r2, 2]] += events[r1:r2, 0]

    for i in range(10):
        frames[i, :, :, :] = frames[i, :, :, :] / np.max(frames[i, :, :, :])

    return frames, label


def create_npy():
    train_filename = 'dvs-cifar10/train/{}.pt'
    test_filename = 'dvs-cifar10/test/{}.pt'
    if not os.path.exists('dvs-cifar10/train'):
        os.mkdir('dvs-cifar10/train')
    if not os.path.exists('dvs-cifar10/test'):
        os.mkdir('dvs-cifar10/test')

    train_test_portion = 0.7

    fns_train = gather_addr('dvs-cifar10', 0, int(train_test_portion * 1000))
    fns_test = gather_addr('dvs-cifar10', int(train_test_portion * 1000), 1000)

    print("processing training data...")
    key = -1
    for file_d in fns_train:
        if key % 100 == 0:
            print("\r\tTrain data {:.2f}% complete\t\t".format(key / train_test_portion / 100), end='')
        frames, labels = events_to_frames(file_d, dt=5000)
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor([labels,])],
                   train_filename.format(key))

    print("\nprocessing testing data...")
    key = -1
    for file_d in fns_test:
        if key % 100 == 0:
            print("\r\tTest data {:.2f}% complete\t\t".format(key / (1 - train_test_portion) / 100), end='')
        frames, labels = events_to_frames(file_d, dt=5000)
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor(labels)],
                   test_filename.format(key))
    print('\n')

if __name__ == "__main__":
    create_npy()
