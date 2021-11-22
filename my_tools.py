import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
import seaborn as sns


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    median_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            median_grads.append(p.grad.abs().median())

    width = 0.3
    plt.bar(np.arange(len(max_grads)), max_grads, width, color="c")
    plt.bar(np.arange(len(max_grads)) + width, ave_grads, width, color="b")
    plt.bar(np.arange(len(max_grads)) + 2 * width, median_grads, width, color='r')

    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="r", lw=4)], ['max-gradient', 'mean-gradient', 'median-gradient'])
    plt.show()


def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR * len(dataset)), len(dataset)))
    return trainDataset, valDataset


def paint_smi_matrixs(matrixs, index=0):
    plt.clf()
    b, c, w, h = matrixs.shape
    for i in range(c):
        matrix = matrixs[0, i, :, :].detach().cpu().numpy()
        plt.imshow(matrix)
        plt.colorbar()
        dir = '/p300/graph/matrixs{0}'.format(index)
        if not os.path.exists(dir):
            os.mkdir('/p300/graph/matrixs{0}'.format(index))
        plt.savefig(fname="/p300/graph/matrixs{0}/matrix{1}.png".format(index, str(i)), dpi=400)
        plt.close()


def plot_inference(precount, count):
    precount = precount.cpu()
    count = count.cpu()
    plt.plot(precount, color='blue')
    plt.plot(count, color='red')
    plt.savefig(fname="/p300/plot/inference.jpg", dpi=400)


def density_map(maps,count,index):
    plt.clf()
    map=maps.detach().cpu().numpy().reshape(1,64)
    sns.set()
    fig = plt.figure(figsize=(64, 4))
    sns_plot = sns.heatmap(map, xticklabels=False, cbar=False, cmap='Greens')
    plt.savefig(fname="/p300/density_map/{0}_{1}.png".format(index, str(count)), dpi=400)
    plt.close()



