import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dvscifar_dataloader import DVSCifar10
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')


def main():
    global arg
    arg = parser.parse_args()

    train_filename = "dvs-cifar10/train/"
    test_filename = "dvs-cifar10/test/"

    train_loader = DataLoader(DVSCifar10(train_filename),
                              batch_size=arg.batch_size,
                              shuffle=False)
    
    test_loader = DataLoader(DVSCifar10(test_filename),
                             batch_size=arg.batch_size,
                             shuffle=False)


    # Model
    model = Spatial_CNN(nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=None
    )
    
    # Training
    model.run()


class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.test_video = test_video
        self.image_size = np.array([0, 0])

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        # build model
        self.model = spiking_resnet_18(self.image_size, self.batch_size, nb_classes=10, channel=2).cuda()
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                # self.best_prec1 = checkpoint['best_prec1']
                self.best_prec1 = 30
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def get_imagesize(self):
        progress = tqdm(self.train_loader)
        for i, (data0, label) in enumerate(progress):
            batch_size, window, ch, w, h = data0.size()
            self.image_size = np.array([w, h])
            self.batch_size = batch_size
            break

    def run(self):
        self.get_imagesize()
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            if self.epoch % 10 == 9:
                is_best = True
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, is_best, 'record/spatial/checkpoint.pth.tar', 'record/spatial/model_best.pth.tar')
            if self.epoch % 100 == 99:
                prec1, val_loss = self.validate_1epoch()
                is_best = prec1 > self.best_prec1
                # save model
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()
                },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data0, label) in enumerate(progress):
            # measure data loading time
            data_time.update(time.time() - end)
            output = self.model(data0.cuda())
            loss = self.criterion(output, label.view((-1)).cuda())
            # --- show some event images
            # plt.figure()
            # for k in range(32):
            #     plt.subplot(6, 6, k + 1)
            #     plt.imshow(data0[k, 0, 0, :, :].numpy())
            #     plt.title(label.numpy().transpose()[0, k])
            # plt.show()
                # # for j in range(10):
                # cv2.imshow("frame", )
                # cv2.waitKey(100)
            # print(label.numpy().transpose())
            # --- measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label.cuda(), topk=(1, 5))
            losses.update(loss.item(), data0.size(0))
            top1.update(prec1.item(), data0.size(0))
            top5.update(prec5.item(), data0.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            loss.backward()
            # This line is used to prevent the vanishing / exploding gradient problem
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (data0, label) in enumerate(progress):
                output = self.model(data0.cuda)
                # compute output
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # Calculate pred acc
                prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
                top1.update(prec1.item(), data0.size(0))
                top5.update(prec5.item(), data0.size(0))

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                }
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return True


if __name__ == '__main__':
    main()