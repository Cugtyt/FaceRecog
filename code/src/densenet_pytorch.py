"""
Debsenet pytorch code.

Use densenet to implement face recognition algo.
"""
from torch import nn, cat
from functools import reduce
from operator import mul


class DenseBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outplanes * 2, outplanes * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(outplanes * 3, outplanes * 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes * 2)
        self.bn3 = nn.BatchNorm2d(outplanes * 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, steps: int):
        print('dense')
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = cat([x, x1], 1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x3 = cat([x, x1, x2], 1)
        if steps == 2:
            return x3
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        out = cat([x, x1, x2, x3], 1)
        return out


class TransitionBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        print('transition')
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, classes=10):
        super(DenseNet, self).__init__()
        # self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout()
        self.dense1 = DenseBlock(3, 32)
        self.dense2 = DenseBlock(32, 64)
        self.trans1 = TransitionBlock(99, 32)
        self.trans2 = TransitionBlock(512, 128)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=0)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = self.dense1(x, 3)
        # x = self.pool(x)
        x = self.trans1(x)
        x = self.dense2(x, 2)
        # x = self.pool(x)
        x = self.trans2(x)
        x = x.view(-1, reduce(mul, x.size()[1:]))
        x = self.relu(self.fc1(x))
        x = self.softmax(self.dropout(self.fc2(x)))
        return x