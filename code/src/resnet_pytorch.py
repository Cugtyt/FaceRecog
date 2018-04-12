"""
Resnet pytorch code.

Use resnet to implement face recognition algo.
"""
from torch import nn
from functools import reduce
from operator import mul


class IdentityBLock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(IdentityBLock, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.bn(out)
#         out = self.relu(out)

        out += residual
        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, outplanes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, classes=10):
        super(ResNet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout()
        self.id = IdentityBLock(32, 32)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=0)
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.id(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, reduce(mul, x.size()[1:]))
#         x = self.relu(self.fc1(x))
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.softmax(self.fc2(x))
#         x = self.softmax(self.dropout(self.fc2(x)))

        return x
