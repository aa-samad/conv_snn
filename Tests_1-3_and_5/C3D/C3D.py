import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_class = 20
batch_size = 100
learning_rate = 1e-3
num_epochs = 10  # max epoch


# cnn_layer(in_planes(channels), out_planes(channels), stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]         # conv layers input image shape (+ last output shape)
# fc layer
cfg_fc = [128, num_class]              # linear layers output


class C3D(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.fc6 = nn.Linear(256 * 2 * 3 * 3, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc8 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, input, time_window=10):
        x = input[:, :, :, :].float()
        x = x.view(-1, 1, time_window,  28, 28)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = x.view(-1, 256 * 2 * 3 * 3)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

