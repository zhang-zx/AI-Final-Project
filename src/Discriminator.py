import torch
import torch.nn as nn
import torch.nn.functional as F
from util import normal_init

class discriminator(nn.Module):
    """docstring for discriminator"""
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        # self.fc1_1_bn = nn.BatchNorm1d(1024)
        self.fc1_2 = nn.Linear(10, 1024)
        # self.fc1_2_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
        self.weight_init(mean=0, std=0.02)
        self.cuda()
        
    def weight_init(self, mean, std):
        for module in self._modules:
            normal_init(self._modules[module], mean, std)

    def forward(self, input_data, label):
        x = F.leaky_relu(self.fc1_1(input_data), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x