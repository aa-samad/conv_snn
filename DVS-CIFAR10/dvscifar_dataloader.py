import struct
import numpy as np
import scipy.misc
import h5py
import glob
from events_timeslices import *
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import os


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
        self.data, self.labels = torch.load(self.root)
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # print(len(data), len(target), self.transform, self.target_transform)

        return data, target.long()

    def __len__(self):
        return len(self.data)


class SequenceGenerator(object):
    def __init__(self,
                 filename = 'DVS-CIFAR10/dvs_cifar10_events.hdf5',
                 group = 'train',
                 batch_size = 32,
                 chunk_size = 6,
                 ds = 2,
                 size = [2, 128, 128],
                 dt = 16000):

        self.group = group
        self.dt = dt
        self.ds = ds
        self.size = size
        f = h5py.File(filename, 'r', swmr=True, libver="latest")
        self.stats = f['stats']
        self.grp1 = f[group]
        self.num_classes = 10
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def reset(self):
        self.i = 0

    def next(self, offset = 0):
        dat,lab = next_1ofeach(
                self.grp1,
                T = self.chunk_size,
                n_classes = self.num_classes,
                size = self.size,
                ds = self.ds,
                dt = self.dt,
                offset = offset)

        return dat, lab


def gather_gestures_stats(hdf5_grp):
    from collections import Counter
    labels = []
    for d in hdf5_grp:          
        labels += hdf5_grp[d]['labels'][:,0].tolist()
    count = Counter(labels)
    stats = np.array(list(count.values()))
    stats = stats / stats.sum()
    return stats


def aedat_to_events(filename):
    label_filename = filename[:].split('/')[1]
    labels = np.ones((10, 1)) * int(list(mapping.values()).index(label_filename))
    events = []
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True: 
            data_ev_head = f.read(28)
            if len(data_ev_head) == 0:
                break

            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            if eventtype == 1:
                event_bytes = np.frombuffer(f.read(eventnumber * eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1, 2)

                x = (event_bytes[:, 0] >> 17) & 0x00001FFF
                y = (event_bytes[:, 0] >> 2) & 0x00001FFF
                p = (event_bytes[:, 0] >> 1) & 0x00000001
                t = event_bytes[:, 1]
                # print(x, y, p, t)
                # return
                events.append([t, x, y, p])
            else:
                f.read(eventnumber * eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    clipped_events = np.zeros([4, 0],'uint32')
    for l in labels:
        start = np.searchsorted(events[0, :], l[1])
        end = np.searchsorted(events[0, :], l[2])
        clipped_events = np.column_stack([clipped_events, events[:, start:end]])
    return clipped_events.T, labels


def compute_start_time(labels,pad):
    l0 = np.arange(len(labels[:,0]), dtype='int')
    np.random.shuffle(l0)
    label = labels[l0[0],0]
    tbegin = labels[l0[0],1]
    tend = labels[l0[0],2]-pad
    start_time = np.random.randint(tbegin, tend)
    return start_time, label


def next(hdf5_group, stats, batch_size = 32, T = 500, n_classes = 11, ds = 2, size = [2, 128, 128], dt = 1000):
    batch = np.empty([batch_size,T]+size, dtype='float')
    batch_idx = np.arange(len(hdf5_group), dtype='int')
    np.random.shuffle(batch_idx)
    batch_idx = batch_idx[:batch_size]
    batch_idx_l = np.empty(batch_size, dtype= 'int')
    for i, b in (enumerate(batch_idx)):
        dset = hdf5_group[str(b)]
        labels = dset['labels'].value
        cand_batch = -1
        while cand_batch is -1: #catches some mislabeled data
            start_time, label = compute_start_time(labels, pad = 2*T*dt)
            batch_idx_l[i] = label-1
            #print(str(i),str(b),mapping[batch_idx_l[i]], start_time)
            cand_batch = get_event_slice(dset['time'].value, dset['data'], start_time, T, ds=ds, size=size, dt=dt)
        batch[i] = cand_batch
    return batch, expand_targets(one_hot(batch_idx_l, n_classes), T).astype('float')


def next_1ofeach(hdf5_group, T = 500, n_classes = 11, ds = 2, size = [2, 128, 128], dt = 1000, offset = 0):
    batch_1of_each = {k:range(len(v['labels'].value)) for k,v in hdf5_group.items()}
    batch_size = int(np.sum([len(v) for v in batch_1of_each.values()]))
    batch = np.empty([batch_size,T] + size, dtype='float')
    batch_idx_l = np.empty(batch_size, dtype= 'int')
    i = 0
    for b,v in batch_1of_each.items():
        for l0 in v:
            dset = hdf5_group[str(b)]
            labels = dset['labels'].value
            label = labels[l0,0]
            batch_idx_l[i] = label-1
            start_time = labels[l0,1] + offset*dt
            #print(str(i),str(b),mapping[batch_idx_l[i]], start_time)
            batch[i] = get_event_slice(dset['time'].value, dset['data'], start_time, T, ds=ds, size=size, dt=dt)
            i += 1
    return batch, expand_targets(one_hot(batch_idx_l, n_classes), T).astype('float')


def get_event_slice(times, addrs, start_time, T, size = [128, 128], ds = 1, dt = 1000):
    try:
        idx_beg = find_first(times, start_time)
        idx_end = find_first(times[idx_beg:], start_time+T*dt)+idx_beg
        return chunk_evs_pol(times[idx_beg:idx_end], addrs[idx_beg:idx_end], deltat=dt, chunk_size=T, size = size, ds = ds)
    except IndexError:
        print("Empty batch found, returning -1")
        return -1


def gather_aedat(directory, start_id, end_id, filename_prefix = 'cifar10'):
    import glob
    fns = []
    for class0 in mapping.keys():
        for i in range(start_id, end_id):
            search_mask = directory + '/' + mapping[class0] + '/' + filename_prefix + "_" + mapping[class0] + "_" + "{0:02d}".format(i) + '*.aedat'
            # search mask example: 
            glob_out = glob.glob(search_mask)
            if len(glob_out)>0:
                fns+=glob_out
    return fns


def create_events_hdf5():
    fns_train = gather_aedat('DVS-CIFAR10', 0, 700)
    fns_test = gather_aedat('DVS-CIFAR10', 700, 1000)

    with h5py.File('DVS-CIFAR10/dvs_cifar10_events.hdf5', 'w') as f:
        f.clear()

        print("processing training data...")
        key = 0
        train_grp = f.create_group('train')
        for file_d in fns_train:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = train_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:, 0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:, 1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        print("processing testing data...")
        key = 0
        test_grp = f.create_group('test')
        for file_d in fns_test:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = test_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        stats =  gather_gestures_stats(train_grp)
        f.create_dataset('stats',stats.shape, dtype = stats.dtype)
        f['stats'][:] = stats


def create_data(batch_size = 64, chunk_size = 6, size = [2, 128, 128], ds = 2, dt = 16000):
    strain = SequenceGenerator(group='train', batch_size = batch_size, chunk_size = chunk_size, size = size, ds = ds, dt= dt)
    stest = SequenceGenerator(group='test', batch_size = batch_size, chunk_size = chunk_size, size = size, ds = ds, dt= dt)
    return strain, stest


def plot_gestures_imshow(images, labels, nim=10, avg=50, do1h = True, transpose=False):
    import pylab as plt
    plt.figure(figsize = [nim+2,16])
    import matplotlib.gridspec as gridspec
    if not transpose:
        gs = gridspec.GridSpec(images.shape[1]//avg, nim)
    else:
        gs = gridspec.GridSpec(nim, images.shape[1]//avg)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=0.95, wspace=.0, hspace=.04)
    if do1h:
        categories = labels.argmax(axis=1)
    else:
        categories = labels
    s=[]
    for j in range(nim):
         for i in range(images.shape[1]//avg):
             if not transpose:
                 ax = plt.subplot(gs[i, j])
             else:
                 ax = plt.subplot(gs[j, i])
             plt.imshow(images[j,i * avg: (i * avg + avg), 0, :, :].sum(axis=0).T)
             plt.xticks([])
             if i==0:  plt.title(mapping[labels[0, j].argmax()], fontsize=10)
             plt.yticks([])
             plt.gray()
         s.append(images[j].sum())
    print(s)
    #return images,labels



    # pass
if __name__ == "__main__":
    create_events_hdf5()
    d_train, d_test = create_data(chunk_size=112, size = [2, 128, 128], ds = 2, dt = 10000)
    x_train, y_train = d_train.next()
    x_test,  y_test  = d_test.next()
    x_train = x_train[:,:,0,:,:].astype('uint8')
    y_train = y_train.transpose((1,0,2)).astype('uint8')
    x_test = x_test[:,:,0,:,:].astype('uint8')
    y_test = y_test.transpose((1, 0, 2)).astype('uint8')
    torch.save([torch.Tensor(x_train[:, :108].reshape(-1, 6, 128, 128)),
                torch.Tensor(np.argmax(y_train[:, :108].reshape(-1, 6, 11)[:, -1], axis=1))],
                train_filename)

    torch.save([torch.Tensor(x_test[:, :108].reshape(-1, 6, 128, 128)),
                torch.Tensor(np.argmax(y_test[:, :108].reshape(-1, 6, 11)[:, -1], axis=1))],
                test_filename)
