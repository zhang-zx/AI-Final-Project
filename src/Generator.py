import torch
import torch.nn as nn
import torch.nn.functional as F
from util import normal_init

class generator(nn.Module):
    """docstring for generator"""
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)
        self.weight_init(mean=0, std=0.02)
        self.cuda()

    def weight_init(self, mean, std):
        for module in self._modules:
            normal_init(self._modules[module], mean, std)

    def forward(self, input_data, label):
        z = F.relu(self.fc1_1_bn(self.fc1_1(input_data)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        z = torch.cat([z, y], 1)
        z = F.relu(self.fc2_bn(self.fc2(z)))
        z = F.relu(self.fc3_bn(self.fc3(z)))
        z = F.tanh(self.fc4(z))
        return z