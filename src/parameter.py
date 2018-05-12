import torch
from torch.autograd import Variable


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 500


temp_z = torch.rand(10, 100)
fixed_z = temp_z
fixed_y = torch.zeros(10, 1)

for i in range(9):
    fixed_z = torch.cat([fixed_z, temp_z], 0)
    temp = torch.ones(10,1) + i
    fixed_y = torch.cat([fixed_y, temp], 0)
    

fixed_z = Variable(fixed_z.cuda(), volatile=True)
fixed_y_label = torch.zeros(100, 10)
fixed_y_label.scatter_(1, fixed_y.type(torch.LongTensor), 1)
fixed_y_label = Variable(fixed_y_label.cuda(), volatile=True)