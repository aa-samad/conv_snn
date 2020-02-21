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
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long()

    def __len__(self):
        return len(os.listdir(self.root))


def gather_addr(directory, start_id, end_id):
    import glob
    fns = []
    for class0 in mapping.keys():
        for i in range(start_id, end_id):
            search_mask = directory + '/' + mapping[class0] + '/' + "{0:02d}".format(i) + '*.mat'
            # search mask example: 
            glob_out = glob.glob(search_mask)
            if len(glob_out) > 0:
                fns += glob_out
    return fns


def events_to_frames(filename, dt):
    label_filename = filename[:].split('/')[1]
    label = int(list(mapping.values()).index(label_filename))
    frames = np.zeros((10, 2, 128, 128))
    events = loadmat('dvs-cifar10/airplane/0.mat')['out1']
    # TODO: next stage = speed inv time surface
    for i in range(10):
        frames[i, events[i * dt: (i+1) * dt, 3], events[i * dt: (i+1) * dt, 1], events[i * dt: (i+1) * dt, 2]] = 1
    return frames, label


def create_npy():
    train_filename = 'dvs-cifar10/train/{}.pt'
    test_filename = 'dvs-cifar10/test/{}.pt'
    if not os.path.exists('dvs-cifar10/train'):
        os.mkdir('dvs-cifar10/train')
    if not os.path.exists('dvs-cifar10/test'):
        os.mkdir('dvs-cifar10/test')

    fns_train = gather_addr('dvs-cifar10', 0, 700)
    fns_test = gather_addr('dvs-cifar10', 700, 1000)

    print("processing training data...")
    key = -1
    # x_train = []
    # y_train = []
    for file_d in fns_train:
        if key % 100 == 0:
            print(key)
        frames, labels = events_to_frames(file_d, dt=10000)
        # x_train.append(frames)
        # y_train.append(labels)
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor([labels,])],
                   train_filename.format(key))
        # x_train = []
        # y_train = []

    print("processing testing data...")
    key = -1
    # x_test = []
    # y_test = []
    for file_d in fns_test:
        if key % 100 == 0:
            print(key)
        frames, labels = events_to_frames(file_d, dt=10000)
        # x_test.append(frames)
        # y_test.append(labels)
        key += 1
        torch.save([torch.Tensor(frames), torch.Tensor(labels)],
                   test_filename.format(key))
        # x_test = []
        # y_test = []




    # pass
if __name__ == "__main__":
    create_npy()
