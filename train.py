'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

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
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--level', default=15, type=int, help='image quantization level')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = ResNet50(level=args.level)


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
encoder = encoder(level=args.level)
LSPGA = LSPGA(model=net,epsilon=0.032,k=args.level,delta=1.2,xi=1.5,step=7,criterion=criterion,encoder=encoder)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        channel0,channel1,channel2=inputs.numpy()[:,0,:,:],inputs.numpy()[:,1,:,:],inputs.numpy()[:,2,:,:]
        channel0,channel1,channel2 = encoder.tempencoding(channel0),encoder.tempencoding(channel1),encoder.tempencoding(channel2)
        channel0, channel1, channel2 = torch.Tensor(channel0),torch.Tensor(channel1),torch.Tensor(channel2)
        if use_cuda:
            channel0, channel1, channel2,targets = channel0.cuda(), channel1.cuda(), channel2.cuda(),targets.cuda()
        optimizer.zero_grad()
        channel0, channel1, channel2, targets = Variable(channel0),Variable(channel1),Variable(channel2), Variable(targets)
        outputs = net(channel0, channel1, channel2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        channel0,channel1,channel2=inputs.numpy()[:,0,:,:],inputs.numpy()[:,1,:,:],inputs.numpy()[:,2,:,:]
        channel0,channel1,channel2 = encoder.tempencoding(channel0),encoder.tempencoding(channel1),encoder.tempencoding(channel2)
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

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def advtrain(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        channel0,channel1,channel2 = LSPGA.attackthreechannel(inputs,targets)
        channel0, channel1, channel2 = torch.Tensor(channel0),torch.Tensor(channel1),torch.Tensor(channel2)
        if use_cuda:
            channel0, channel1, channel2,targets = channel0.cuda(), channel1.cuda(), channel2.cuda(),targets.cuda()
        optimizer.zero_grad()
        channel0, channel1, channel2, targets = Variable(channel0),Variable(channel1),Variable(channel2), Variable(targets)
        outputs = net(channel0, channel1, channel2)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
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

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)