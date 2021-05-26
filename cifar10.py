# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 20:17
# @Author  : Nin
# @File    : cifar10.py
# @Software: PyCharm
import torch
import time
import torchvision
from tqdm import tqdm, tqdm_notebook
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
from torchsummary import summary
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler

path_model = "./checkpoint/"

BATCH_SIZE = 512
NUM_WORKERS =4
# transform
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    # transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])
writer = SummaryWriter('AlexNet')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sampler = torch.utils.data.SubsetRandomSampler(indices=list(range(1000)))
# 创建训练集
# root -- 数据存放的目录
# train -- 明确是否是训练集
# download -- 是否需要下载
# transform -- 转换器，将数据集进行转换
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

# 创建测试集
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)
# 创建训练/测试加载器，
# trainset/testset -- 数据集
# batch_size -- 不解释
# shuffle -- 是否打乱顺序
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=True)

# 类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 学习率
LR = 0.005
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2),
            # 论文中kernel_size = 11,stride = 4,padding = 2
            nn.ReLU(),
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # IN : 256*4*4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),  # 论文中为kernel_size = 3,stride = 2
            # OUT : 384*2*2
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=10),
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.view(-1, 384 * 2 * 2)
        x = self.linear(x)
        return x

net = AlexNet()
# net.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
# optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.95, weight_decay=5e-4)

def train(model, criterion, optimizer, trainloader, epochs=5, log_interval=50):
    print('----- Train Start -----')
    start_time = time.time()

    for epoch in range(epochs):
        train_iter = tqdm(trainloader, maxinterval=10, mininterval=2, ncols=80,
                          bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',
                          smoothing=0.1)
        running_loss = 0.0
        for step, (batch_x, batch_y) in enumerate(train_iter):
            # batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            output = model(batch_x)

            optimizer.zero_grad()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            train_iter.set_description('epoch %d' %epoch)
            train_iter.set_postfix_str('loss={:^7.3f}'.format(running_loss))

        lr_scheduler.step()  # 更新


        # ----------------------------------------- #
        # save_state
        # ----------------------------------------- #
        print('===> Saving models...')
        state = {
            'state': net.state_dict(),
            'epoch': epoch  # 将epoch一并保存
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('./checkpoint')
        torch.save(model.state_dict(),'./checkpoint/Epoch%d-Loss%.4f.pth'%(epoch,running_loss))

    end_time = time.time()
    print('----- Train Finished -----time: %.4f ' % (end_time - start_time))

def test(model, testloader):
    print('------ Test Start -----')

    correct = 0
    total = 0

    with torch.no_grad():
        for test_x, test_y in testloader:
            images, labels = test_x.cuda(), test_y.cuda()
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Accuracy of the network is: %.4f %%' % accuracy)
    return accuracy
if __name__ == '__main__':
    print(net)
    print("start train!")
    optimizer = optim.Adam(net.parameters(),lr=1e-3)
    lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.9)

    train(net, criterion, optimizer, trainloader, epochs=50)
    test(net, testloader)