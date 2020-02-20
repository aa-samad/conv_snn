import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
thresh = 0.5  # neuronal threshold
lens = 0.5/3  # hyper-parameters of approximate function
decay = 0.8  # decay constants

# define approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = torch.exp(-(input - thresh) **2/(2 * lens ** 2))/((2 * lens * 3.141592653589793) ** 0.5) 
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


def mem_update(x, mem, spike):
    mem = mem * decay * (1. - spike) + x
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike



class SpikingBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, image_size, batch_size, stride=1, downsample=None):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        # self.drop1 = nn.Dropout(0.6)
        self.drop2 = nn.Dropout(0.2)
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        w, h = image_size 
                

        
    def forward(self, x, c1_mem, c1_spike, c2_mem, c2_spike):
        residual = x
        out = self.conv1(x)
        c1_mem, c1_spike = mem_update(out, c1_mem, c1_spike)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        c2_mem, c2_spike = mem_update(out, c2_mem, c2_spike)
        c2_spike = self.drop2(c2_spike)
        return c2_spike, c1_mem, c1_spike, c2_mem, c2_spike


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, image_size, batch_size, nb_classes=101, channel=20):
        self.inplanes = 64
        super(SpikingResNet, self).__init__()
        self.nb_classes = nb_classes
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,   
                               bias=False)
        #self.drop = nn.Dropout(0.2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.layer_num = layers
        self.size_devide = np.array([4, 4, 4, 4])
        self.planes = [64, 64, 64, 64]
        self._make_layer(block, 64, layers[0], image_size//4, batch_size)     # //2 due to stride=2 conv1_custom & avgpool
        self._make_layer(block, 64, layers[1], image_size//4, batch_size, stride=1)  # //2 due to stride=2 conv1_custom & avgpool
        self._make_layer(block, 64, layers[2], image_size//4, batch_size, stride=1)  # //4 due to stride=2 conv1_custom & avgpool & layer2 
        self._make_layer(block, 64, layers[3], image_size//4, batch_size, stride=1)  # //8 due to stride=2 conv1_custom & avgpool & layer2 & layer3
        self.avgpool2 = nn.AvgPool2d(7)
        # self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        self.fc_custom = nn.Linear(64 * 3 * 8 * 8, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 1)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(1. / n))

    def _make_layer(self, block, planes, blocks, image_size, batch_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False))

        # layers = []
        self.layers.append(block(self.inplanes, planes, image_size, batch_size, stride, downsample).cuda())
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.layers.append(block(self.inplanes, planes, image_size//stride, batch_size).cuda())


    def forward(self, input):
        time_window, batch_size, ch, w, h = input.size()
        # ---- creating membrane potential and spike memory
        c_mem = c_spike = torch.zeros(batch_size, 64, w // 2, h // 2, device=device)  # //2 due to stride=2
        c2_spike, c2_mem, c1_spike, c1_mem = [], [], [], []
        for i in range(len(self.layer_num)):
            d = self.size_devide[i]
            for j in range(self.layer_num[i]):
                c1_mem.append(torch.zeros(batch_size, self.planes[i], w//d, h//d, device=device))
                c1_spike.append(torch.zeros(batch_size, self.planes[i], w//d, h//d, device=device))
                c2_mem.append(torch.zeros(batch_size, self.planes[i], w//d, h//d, device=device))
                c2_spike.append(torch.zeros(batch_size, self.planes[i], w//d, h//d, device=device))
        fc_sumspike = fc_mem = fc_spike = torch.zeros(batch_size, self.nb_classes, device=device)  # //2 due to stride=2

        for step in range(time_window):
            x = input[step, :, :, :, :]
            x = self.conv1_custom(x)
            c_mem, c_spike = mem_update(x, c_mem, c_spike)
            # --------- blocks
            x = self.avgpool1(c_spike)
            for i in range(len(self.layers)):
                x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i] = \
                self.layers[i](x, c1_mem[i], c1_spike[i], c2_mem[i], c2_spike[i])
            # ---------
            x = torch.cat(c2_spike[-1], c2_spike[-2], c2_spike[-3], dim=1)
            x = self.avgpool2(x)
            x = x.view(x.size(0), -1)
            out = self.fc_custom(x)
            fc_mem, fc_spike = mem_update(out, fc_mem, fc_spike)
            fc_sumspike += fc_spike
        fc_sumspike = fc_sumspike / time_window 
        return fc_sumspike



def spiking_resnet_18(image_size, batch_size, channel=3, **kwargs):
    model = SpikingResNet(SpikingBasicBlock, [2, 2, 2, 2], image_size, batch_size, nb_classes=101, channel=channel, **kwargs)
    return model


def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict

     