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
            theta1 = 0
            theta2 = 360
            for k in range(10):
                if k == 0 or k == 9:
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)  # giving empty frame (this line does nothing!)
                else:
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta1 + random.randrange(0, 360, 36), 1.0)  # rotate counter clock-wise
                # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), k * 36, 1.0)  # rotate counter clock-wise
                # M = np.float32([[1 - 0.1 * k, 0, 0], [0, 1 - 0.1 * k, 0]])     # zoom out
                # M = np.float32([[1 - 0.05 * k, 0, 0], [0, 1 - 0.05 * k, 0]])  # zoom out less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2, k, :, :] = torch.from_numpy(dst)
                labels2[j * 2] = labels[j]
            for k in range(1, 11):
                if k == 0 or k == 9:
                    images2[j * 2 + 1, k, :, :] = torch.from_numpy(img0)
                else:
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta2 - random.randrange(0, 360, 36), 1.0)  # rotate clock-wise
                # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 360 - k * 36, 1.0)  # rotate clock-wise
                # M = np.float32([[0.1 * k, 0, 0], [0, 0.1 * k, 0]])    # zoom in
                # M = np.float32([[0.5 + 0.05 * k, 0, 0], [0, 0.5 + 0.05 * k, 0]])  # zoom in less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2 + 1, k - 1, :, :] = torch.from_numpy(dst)
                labels2[j * 2 + 1] = labels[j] + 10
        # ----
        images2 = images2.numpy().reshape([-1, 10, 28, 28, 1])
        labels2 = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1).numpy()
        images3.append(images2)
        labels3.append(labels2)
    images2 = np.array(images3).reshape((-1, 10, 28, 28, 1))
    labels2 = np.array(labels3).reshape((-1, 20))
    seq.fit(images2, labels2, batch_size=128, epochs=1, validation_split=0.05)

    # ============== Test ================
    cm = np.zeros((20, 20), dtype=np.int32)
    correct = 0
    total = 0
    images3 = []
    labels3 = []
    print("test data len = {}".format(len(test_loader)))
    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx % 10 == 0:
            print("\rstep {}/{} of creating test images".format(batch_idx, len(test_loader)), end='')
        if batch_idx == len(test_loader) - 1:
            print()
        images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
        labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
        for j in range(images.shape[0]):
            img0 = images[j, 0, :, :].numpy()
            rows, cols = img0.shape
            theta1 = 0
            theta2 = 360
            for k in range(10):
                if k == 0 or k == 9:
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)
                else:
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta1 + random.randrange(0, 360, 36), 1.0)  # rotate counter clock-wise
                # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), k * 36, 1.0)  # rotate counter clock-wise
                # M = np.float32([[1 - 0.1 * k, 0, 0], [0, 1 - 0.1 * k, 0]])     # zoom out
                # M = np.float32([[1 - 0.05 * k, 0, 0], [0, 1 - 0.05 * k, 0]])     # zoom out less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2, k, :, :] = torch.from_numpy(dst)
                labels2[j * 2] = labels[j]
            for k in range(1, 11):
                if k == 0 or k == 9:
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)  # giving empty frame (this line does nothing!)
                else:
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta2 - random.randrange(0, 360, 36), 1.0)  # rotate clock-wise
                # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 360 - k * 36, 1.0)  # rotate clock-wise
                # M = np.float32([[0.1 * k, 0, 0], [0, 0.1 * k, 0]])    # zoom in
                # M = np.float32([[0.5 + 0.05 * k, 0, 0], [0, 0.5 + 0.05 * k, 0]])    # zoom in less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2 + 1, k - 1, :, :] = torch.from_numpy(dst)
                labels2[j * 2 + 1] = labels[j] + 10
                # ----
        images2 = images2.numpy().reshape([-1, 10, 28, 28, 1])
        labels2 = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1).numpy()
        images3.append(images2)
        labels3.append(labels2)
    images2 = np.array(images3).reshape((-1, 10, 28, 28, 1))
    labels2 = np.array(labels3).reshape((-1, 20))

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
    class_names =['0_ccw', '1_ccw', '2_ccw', '3_ccw', '4_ccw',
                 '5_ccw', '6_ccw', '7_ccw', '8_ccw', '9_ccw',
                 '0_cw', '1_cw', '2_cw', '3_cw', '4_cw',
                 '5_cw', '6_cw', '7_cw', '8_cw', '9_cw']
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



