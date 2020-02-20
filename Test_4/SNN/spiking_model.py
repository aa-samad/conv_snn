import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
thresh = 0.5  # neuronal threshold
lens = 0.5/3  # hyper-parameters of approximate function
decay = 0.5  # decay constants
num_classes = 10
batch_size = 100
learning_rate = 5e-4
num_epochs = 20  # max epoch


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
        # temp = abs(input - thresh) < lens
        temp = torch.exp(-(input - thresh) **2/(2 * lens ** 2))/((2 * lens * 3.141592653589793) ** 0.5)
        return grad_input * temp.float()


act_fun = ActFun.apply
# membrane potential update


def mem_update(ops, x, mem, spike):
    mem = mem * decay + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike


# cnn_layer(in_planes(channels), out_planes(channels), kernel_size, stride, padding)
cfg_cnn = [(1, 48, 3, 1, 1),
           (48, 48, 3, 1, 1),
           (48, 64, 3, 1, 1),
           (64, 64, 3, 1, 1),]
# kernel size
# cnn output shapes (conv1, conv2, fc1 input)
cfg_kernel = [28, 27, 26, 13, 6]         # conv layers input image shape (+ last output shape)
# fc layer
cfg_fc = [128, 10]              # linear layers output


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[2]
        self.conv3  = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, kernel_size, stride, padding = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-2] * cfg_kernel[-2] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window=10):
        # convolutional layers membrane potential and spike memory
        c1_mem = c1_spike = torch.zeros(batch_size * 2, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size * 2, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)
        
        # linear layers membrane potential and spike memory
        c3_mem = c3_spike = torch.zeros(batch_size * 2, cfg_cnn[2][1], cfg_kernel[2], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size * 2, cfg_cnn[2][2], cfg_kernel[3], cfg_kernel[3], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size * 2, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size * 2, cfg_fc[1], device=device)

        for step in range(time_window):  # simulation time steps
            x = input[:, step: step + 1, :, :]

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2, stride =1, padding=0)

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)
            x = F.avg_pool2d(c2_spike, 2, stride =1, padding=0)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)
            x = F.avg_pool2d(c3_spike, 2)

            x = x.view(batch_size * 2, -1)      # flatten

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs


