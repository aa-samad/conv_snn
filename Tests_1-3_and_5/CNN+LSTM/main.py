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

snn = SCNN()
snn.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    running_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
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
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)    
                else:
                    # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), k * 36, 1.0)  # rotate counter clock-wise
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta1 + random.randrange(0,360,36), 1.0)  # rotate counter clock-wise
                    # M = np.float32([[1 - 0.05 * k, 0, 0], [0, 1 - 0.05 * k, 0]])  # zoom out less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2, k, :, :] = torch.from_numpy(dst)
                labels2[j * 2] = labels[j]
            for k in range(1, 11):
                if k == 0 or k == 9:
                    images2[j * 2, k, :, :] = torch.from_numpy(img0)    
                else:
                    # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 360 - k * 36, 1.0)  # rotate clock-wise
                    M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta2 - random.randrange(0,360,36), 1.0)  # rotate counter clock-wise
                    # M = np.float32([[0.5 + 0.05 * k, 0, 0], [0, 0.5 + 0.05 * k, 0]])  # zoom in less aggressive
                    dst = cv2.warpAffine(img0, M, (cols, rows))
                    images2[j * 2 + 1, k - 1, :, :] = torch.from_numpy(dst)
                labels2[j * 2 + 1] = labels[j] + 10
        # ----

        snn.zero_grad()
        optimizer.zero_grad()

        images2 = images2.float().to(device)
        outputs = snn(images2)
        labels_ = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1)
        loss = criterion(outputs.cpu(), labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
             print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                    %( epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, running_loss))
             running_loss = 0
             print('Time elasped:', time.time()-start_time)
    correct = 0
    total = 0
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    cm = np.zeros((20, 20), dtype=np.int32)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images2 = torch.empty((images.shape[0] * 2, 10, images.shape[2], images.shape[3]))
            labels2 = torch.empty((images.shape[0] * 2), dtype=torch.int64)
            for j in range(images.shape[0]):
                img0 = images[j, 0, :, :].numpy()
                rows, cols = img0.shape
                for k in range(10):
                    if k == 0 or k == 9:
                        images2[j * 2, k, :, :] = torch.from_numpy(img0)    
                    else:
                        # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), k * 36, 1.0)  # rotate counter clock-wise
                        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta1 + random.randrange(0,360,36), 1.0)  # rotate counter clock-wise
                        # M = np.float32([[1 - 0.05 * k, 0, 0], [0, 1 - 0.05 * k, 0]])     # zoom out less aggressive
                        dst = cv2.warpAffine(img0, M, (cols, rows))
                        images2[j * 2, k, :, :] = torch.from_numpy(dst)
                    labels2[j * 2] = labels[j]
                for k in range(10):
                    if k == 0 or k == 9:
                        images2[j * 2, k, :, :] = torch.from_numpy(img0)    
                    else:
                        # M = cv2.getRotationMatrix2D((rows / 2, cols / 2), 360 - k * 36, 1.0)  # rotate clock-wise
                        M = cv2.getRotationMatrix2D((rows / 2, cols / 2), theta2 - random.randrange(0,360,36), 1.0)  # rotate counter clock-wise
                        # M = np.float32([[0.5 + 0.05 * k, 0, 0], [0, 0.5 + 0.05 * k, 0]])    # zoom in less aggressive
                        dst = cv2.warpAffine(img0, M, (cols, rows))
                        images2[j * 2 + 1, k - 1, :, :] = torch.from_numpy(dst)
                    labels2[j * 2 + 1] = labels[j] + 10
            inputs = images2.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs)
            labels_ = torch.zeros(batch_size * 2, 20).scatter_(1, labels2.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)

            # ----- showing confussion matrix -----
            cm += confusion_matrix(labels2, predicted)
            # ------ showing some of the predictions -----
            # for image, label in zip(inputs, predicted):
            #     for img0 in image.cpu().numpy():
            #         cv2.imshow('image', img0)
            #         cv2.waitKey(100)
            #     print(label.cpu().numpy())

            total += float(labels2.size(0))
            correct += float(predicted.eq(labels2).sum().item())
            if batch_idx % 100 == 0:
                acc = 100. * float(correct) / float(total)
                print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    class_names = ['0_ccw', '1_ccw', '2_ccw', '3_ccw', '4_ccw',
             '5_ccw', '6_ccw', '7_ccw', '8_ccw', '9_ccw',
             '0_cw', '1_cw', '2_cw', '3_cw', '4_cw',
             '5_cw', '6_cw', '7_cw', '8_cw', '9_cw']
    plot_confusion_matrix(cm, class_names)
    print('Iters:', epoch, '\n\n\n')
    print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    if epoch % 5 == 0:
        print(acc)
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt' + names + '.t7')
        best_acc = acc



