import os, time
# import pickle
# import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from Generator import generator
from Discriminator import discriminator
from parameter import *
from util import *

# load data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
trainSet = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)


G = generator()
D = discriminator()
hist = dict()
hist['D_losses'] = list()
hist['G_losses'] = list()
hist['epoch_time'] = list()
hist['total_time'] = 0
dirPrep()

# Binary Cross Entropy loss
BCELoss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))



start_time = time.time()
print("training start!!")

for epoch in range(train_epoch):
    D_losses = list()
    G_losses = list()

    # learning rate decay
    if epoch >= 150:
        if not [ i for i in range(10) if hist['D_losses'][epoch-1-i]<0.130 or hist['D_losses'][epoch-1-i]>0.145 ]:
            G_optimizer.param_groups[0]['lr'] *= 10
            D_optimizer.param_groups[0]['lr'] *= 10
            print("learning rate change by 10!")
        before = hist['D_losses'][epoch-11]
        after = hist['D_losses'][epoch-1]
        if after >= before:
            G_optimizer.param_groups[0]['lr'] *= 0.7
            D_optimizer.param_groups[0]['lr'] *= 0.7
            print("learning rate change by 0.7!")
    elif epoch >= 6:
        before = hist['D_losses'][epoch-6]
        after = hist['D_losses'][epoch-1]
        if after >= before:
            G_optimizer.param_groups[0]['lr'] *= 0.5
            D_optimizer.param_groups[0]['lr'] *= 0.5
            print("learning rate change by 0.5!")
    
    epoch_start_time = time.time()
    for x, y in train_loader:
        D.zero_grad()
        
        batch = x.size()[0]
        
        y_real = torch.ones(batch)
        y_fake = torch.zeros(batch)
        
        y_label = torch.zeros(batch, 10)
        y_label.scatter_(1, y.view(batch, 1), 1)

        x = x.view(-1, 28 * 28)

        x, y_label, y_real, y_fake = Variable(x.cuda()), Variable(y_label.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
        # train with real data
        D_result = D(x, y_label).squeeze()
        D_real_loss = BCELoss(D_result, y_real)
        
        # generate fake data
        z = torch.rand((batch, 100))
        y = (torch.rand(batch, 1) * 10).type(torch.LongTensor)
        y_label = torch.zeros(batch, 10)
        y_label.scatter_(1, y.view(batch, 1), 1)
        

        z, y_label = Variable(z.cuda()), Variable(y_label.cuda())

        G_result = G(z, y_label)
        # train with fake data
        D_result = D(G_result, y_label).squeeze()
        D_fake_loss = BCELoss(D_result, y_fake)

        D_train_loss = D_fake_loss + D_real_loss
        # optimize D
        D_train_loss.backward()
        D_optimizer.step()
        # record the loss of D
        D_losses.append(D_train_loss.data[0])

        # train G
        G.zero_grad()

        z = torch.rand((batch, 100))
        y = (torch.rand(batch, 1) * 10).type(torch.LongTensor)
        y_label = torch.zeros(batch, 10)
        y_label.scatter_(1, y.view(batch, 1), 1)

        z, y_label = Variable(z.cuda()), Variable(y_label.cuda())

        G_result = G(z, y_label)
        D_result = D(G_result, y_label).squeeze()
        G_loss = BCELoss(D_result, y_real)
        # optimize G
        G_loss.backward()
        G_optimizer.step()

        G_losses.append(G_loss.data[0])

    epoch_time = time.time() - epoch_start_time
    hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    hist['G_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    hist['epoch_time'].append(epoch_time)

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, epoch_time, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    pic_path = 'FashionMNIST_results/Fixed_results/FashionMNIST_' + str(epoch + 1) + '.png'
    show_result(G, fixed_z, fixed_y_label, epoch+1, path=pic_path)

total_time = time.time() - start_time
hist['total_time'] = total_time
resultSaver(G, D, hist, train_epoch)