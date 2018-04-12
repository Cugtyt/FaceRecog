"""
Vanilla CNN pytorch code.

Use simple ConvNet to implement face recognition algo.
"""
import torch.nn as nn
from torch.nn.functional import max_pool2d, relu, dropout, softmax
from torch.optim import lr_scheduler
from functools import reduce
from operator import mul


class VaniliaCNN(nn.Module):
    """Implement vanilla ConvNet model."""
    def __init__(self, classes=10):
        super(VaniliaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, 3)
        
        self.conv4 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        
        self.conv6 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv7 = nn.Conv2d(128, 128, 3)
        
        self.fc1 = nn.Linear(128 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, classes)

        
    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = relu(self.conv3(x))
        x = dropout(max_pool2d(x, 2))
        
        x = relu(self.conv4(x))
        x = relu(self.conv5(x))
        x = dropout(max_pool2d(x, 2))
        
        x = relu(self.conv6(x))
        x = relu(self.conv7(x))
        x = max_pool2d(x, 2)
        
        x = x.view(-1, reduce(mul, x.size()[1:]))
        x = relu(self.fc1(x))
        x = softmax(dropout(self.fc2(x)), dim=0)
        return x
        
    # def num_flat_features(self, x):
        # size = x.size()[1:]
        # return reduce(mul, size)


