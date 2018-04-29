import os, time
import matplotlib.pyplot as plt 
plt.switch_backend('agg')


import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable




# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 50

# load data
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
trainSet = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

# testSet = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False)


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

# 10(classes) * 10(pics)
def show_result(epoch_count, show = False, save = True, path = './'):
    G.eval()
    result = G(fixed_z, fixed_y_label)
    G.train()
    x_size = 10
    y_size = 10
    fig, ax = plt.subplots(y_size, x_size, figsize=(5, 5))
    for i, j in itertools.product(range(x_size), range(y_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(100):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(result[k].cpu().data.view(28, 28).numpy(), cmap='gray')
    label = 'Epoch {0}'.format(epoch_count)
    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

def process(hist, show = False, save = True, path = 'Train.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

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

G = generator()
D = discriminator()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)


G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCELoss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# results save folder
if not os.path.isdir('FashionMNIST_results'):
    os.mkdir('FashionMNIST_results')
if not os.path.isdir('FashionMNIST_results/Fixed_results'):
    os.mkdir('FashionMNIST_results/Fixed_results')

hist = {}
hist['D_losses'] = []
hist['G_losses'] = []
hist['epoch_time'] = []
hist['total_time'] = 0


start_time = time.time()
print("training start")
# print('training start at: ' + start_time)

for epoch in range(train_epoch):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 30:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 40:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    
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
    show_result(epoch+1, path=pic_path)

total_time = time.time() - start_time
hist['total_time'] = total_time

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(hist['epoch_time'])), train_epoch, total_time))
print("Training finished! saving the training results")
torch.save(G.state_dict(), "FashionMNIST_results/generator_param.pkl")
torch.save(D.state_dict(), "FashionMNIST_results/discriminator_param.pkl")

with open('FashionMNIST_results/train_hist.pkl', 'wb') as f:
    pickle.dump(hist, f)

process(hist, path='FashionMNIST_results/FashionMNIST_train_hist.png')

images = []

for e in range(train_epoch):
    img_name = 'FashionMNIST_results/Fixed_results/FashionMNIST_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('FashionMNIST_results/generation_animation.gif', images, fps=3)




















