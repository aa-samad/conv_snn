from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from sklearn.metrics import confusion_matrix
from plot_confmat import plot_confusion_matrix
from net_model import*
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import cv2
import numpy as np
import random
import keras

names = 'spiking_model'
data_path = './raw/'  # todo: input your data path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])


for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    # ============== Train ================
    images3 = []
    labels3 = []
    for i, (images, labels) in enumerate(train_loader):
        # if i > 5:
        #     break
        if i % 10 == 0:
            print("\rstep {}/{} of creating train images".format(i, len(train_loader)), end='')
        if i == len(train_loader) - 1:
            print()
        # --- create zoom-in and zoom-out version of each image
        images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
        labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
        for j in range(images.shape[0]):
            img0 = images[j, 0, :, :].numpy()
            rows, cols = img0.shape
            for k in range(10):
                rand1 = random.randint(0, rows//2)
                rand2 = random.randint(0, cols//2)
                images2[j * 2, k, :, :] = torch.from_numpy(img0)
                images2[j * 2, k, rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0 
                labels2[j * 2] = labels[j]
                # cv2.imshow('image', dst)
                # cv2.waitKey(100)
            for k in range(10):
                rand1 = random.randint(0, rows//2)
                rand2 = random.randint(0, cols//2)
                images2[j * 2 + 1, k, :, :] = torch.from_numpy(img0)
                images2[j * 2 + 1, k,rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0 
                labels2[j * 2 + 1] = labels[j]
         # ----
        images2 = images2.numpy().reshape([-1, 10, 28, 28, 1])
        labels2 = torch.zeros(batch_size * 2, 10).scatter_(1, labels2.view(-1, 1), 1).numpy()
        images3.append(images2)
        labels3.append(labels2)
    images2 = np.array(images3).reshape((-1, 10, 28, 28, 1))
    labels2 = np.array(labels3).reshape((-1, 10))
    seq.fit(images2, labels2, batch_size=128, epochs=1, validation_split=0.05)

    # ============== Test ================
    cm = np.zeros((10, 10), dtype=np.int32)
    correct = 0
    total = 0
    images3 = []
    labels3 = []
    print("test data len = {}".format(len(test_loader)))
    for batch_idx, (images, labels) in enumerate(test_loader):
        # if batch_idx > 2:
        #     break
        if batch_idx % 10 == 0:
            print("\rstep {}/{} of creating test images".format(batch_idx, len(test_loader)), end='')
        if batch_idx == len(test_loader) - 1:
            print()
        images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
        labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
        for j in range(images.shape[0]):
            img0 = images[j, 0, :, :].numpy()
            rows, cols = img0.shape
            for k in range(10):
                rand1 = random.randint(0, rows//2)
                rand2 = random.randint(0, cols//2)
                images2[j * 2, k, :, :] = torch.from_numpy(img0)
                images2[j * 2, k, rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0 
                labels2[j * 2] = labels[j]
                # cv2.imshow('image', dst)
                # cv2.waitKey(100)
            for k in range(10):
                rand1 = random.randint(0, rows//2)
                rand2 = random.randint(0, cols//2)
                images2[j * 2 + 1, k, :, :] = torch.from_numpy(img0)
                images2[j * 2 + 1, k, rand1: rand1 + rows//2, rand2: rand2 + cols//2] = 0 
                labels2[j * 2 + 1] = labels[j]   # ----
        images2 = images2.numpy().reshape([-1, 10, 28, 28, 1])
        labels2 = torch.zeros(batch_size * 2, 10).scatter_(1, labels2.view(-1, 1), 1).numpy()
        images3.append(images2)
        labels3.append(labels2)
    images2 = np.array(images3).reshape((-1, 10, 28, 28, 1))
    labels2 = np.array(labels3).reshape((-1, 10))

    predicted = seq.predict(images2)
    # ----- showing confussion matrix -----
    a = np.argmax(predicted, axis=1)
    b = np.argmax(labels2, axis=1)
    cm = confusion_matrix(b, a)
    # ------ showing some of the predictions -----
    # for image, label in zip(inputs, predicted):
    #     for img0 in image.cpu().numpy():
    #         cv2.imshow('image', img0)
    #         cv2.waitKey(100)
    #     print(label.cpu().numpy())
    a = np.argmax(predicted, axis=1) 
    b = np.argmax(labels2, axis=1)
    acc = len(np.where(a == b)[0]) / predicted.shape[0]

    if batch_idx % 100 == 0:
        acc = 100. * acc
        print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    class_names =['0 ', '1 ', '2 ', '3 ', '4 ',
                 '5 ', '6 ', '7 ', '8 ', '9 ',]
        # ['0_zoom_out', '1_zoom_out', '2_zoom_out', '3_zoom_out', '4_zoom_out',
    #          '5_zoom_out', '6_zoom_out', '7_zoom_out', '8_zoom_out', '9_zoom_out',
    #          '0_zoom_in', '1_zoom_in', '2_zoom_in', '3_zoom_in', '4_zoom_in',
    #          '5_zoom_in', '6_zoom_in', '7_zoom_in', '8_zoom_in', '9_zoom_in']
    plot_confusion_matrix(cm, class_names)
    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: {:.3f}'.format(100.0 * acc))
    print("******************************************")
    acc = 100. * acc
    acc_record.append(acc)
    # if epoch % 5 == 0:
    #     print(acc)
    #     state = {
    #         'acc': acc,
    #         'epoch': epoch,
    #         'acc_record': acc_record,
    #     }
        
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # best_acc = acc



