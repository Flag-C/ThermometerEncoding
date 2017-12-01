'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from util import progress_bar
from torch.autograd import Variable
from encoder import encoder
from LSPGA import LSPGA


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--level', default=15, type=int, help='image quantization level')
parser.add_argument('--log',default='them/res50',type=str,help='path of log')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model

print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.t7')
net = checkpoint['net']
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
encoder = encoder(level=args.level)
LSPGA = LSPGA(model=net,epsilon=0.032,k=args.level,delta=1.2,xi=0.01,step=1,criterion=criterion,encoder=encoder)
writer = SummaryWriter(log_dir=args.log)
def advtest():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        channel0, channel1, channel2 = LSPGA.attackthreechannel(inputs, targets)
        channel0, channel1, channel2 = torch.Tensor(channel0),torch.Tensor(channel1),torch.Tensor(channel2)
        if use_cuda:
            channel0, channel1, channel2,targets = channel0.cuda(), channel1.cuda(), channel2.cuda(),targets.cuda()
        channel0, channel1, channel2, targets = Variable(channel0),Variable(channel1),Variable(channel2), Variable(targets)
        outputs = net(channel0, channel1, channel2)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        advc0,advc1,advc2 = (channel[-3:].data.cpu().numpy() for channel in [channel0,channel1,channel2])
        advc0,advc1,advc2 = (encoder.temp2img(advc) for advc in [advc0,advc1,advc2])
        advc0,advc1,advc2 = (torch.Tensor(advc[:,np.newaxis,:,:]) for advc in [advc0,advc1,advc2])
        advimg = torch.cat((advc0,advc1,advc2),dim=1)
        advimg = torchvision.utils.make_grid(advimg)
        writer.add_image('Imagetest', advimg, batch_idx)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == '__main__':
    advtest()