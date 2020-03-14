clc
clear
% a = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
a = {'dog', 'frog', 'horse', 'ship', 'truck'};
for folder0 = 1:10
    for file0 = 0:999
        if mod(file0, 10) == 0
           fprintf('step: %s %d\n', a{folder0}, file0)
        end
        addr = sprintf('C:\\Users\\Ali\\Desktop\\Grid-cell\\SNN\\datasets\\DVS-CIFAR10\\%s\\cifar10_%s_%d.aedat', a{folder0}, a{folder0}, file0);
        out1 = dat2mat(addr);
        save(sprintf('dvs-cifar10\\%s\\%d.mat', a{folder0}, file0), 'out1')
    end
end