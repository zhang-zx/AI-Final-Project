import torch.nn as nn
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
import itertools
import torch
import pickle
import imageio
import os

def dirPrep():
        # results save folder
    if not os.path.isdir('FashionMNIST_results'):
        os.mkdir('FashionMNIST_results')
    if not os.path.isdir('FashionMNIST_results/Fixed_results'):
        os.mkdir('FashionMNIST_results/Fixed_results')

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
        
def show_result(G, fixed_z, fixed_y_label, epoch_count, show = False, save = True, path = './'):
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
        
def resultSaver(G, D, hist, train_epoch):
    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(hist['epoch_time'])), train_epoch, hist['total_time']))
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


def resCon(x, alpha):
    alpha = 1 - alpha
    for x_ in x:
        for i in range(0, 28, 2):
            for j in range(0, 28, 2):
                aver = (x_[i][j] + x_[i][j + 1] + x_[i + 1][j] + x_[i + 1][j + 1]) / 4.0
                x_[i][j] = (1 - alpha) * x_[i][j] + alpha * aver
                x_[i][j + 1] = (1 - alpha) * x_[i][j + 1] + alpha * aver
                x_[i + 1][j] = (1 - alpha) * x_[i + 1][j] + alpha * aver
                x_[i + 1][j + 1] = (1 - alpha) * x_[i + 1][j + 1] + alpha * aver

    return x

