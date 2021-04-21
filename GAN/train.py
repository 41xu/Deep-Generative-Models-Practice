import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--z_dim', type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--output_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=28, help="size of each image dimension")
    parser.add_argument('--input_channel', type=int, default=1)

    opts = parser.parse_args()
    print(opts)

    # data loader
    transform = transforms.Compose([
        transforms.Resize(opts.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.MNIST('../data/mnist', train=True, download=False, transform=transform)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)

    # optimizer


    for epoch in range(opts.n_epoch):
        for i, (img, _) in enumerate(dataloader):
            print(img.shape) # batch_size, channel, height, width




if __name__ == '__main__':
    main()