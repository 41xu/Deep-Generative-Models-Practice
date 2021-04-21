import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_dim=100, output_size=32, output_channel=3):
        """
        :param input_dim: input random noise z's dim, default=100
        :param output_size:
        """
        super(Generator, self).__init__()

        def block(in_dim, out_dim, normalize=True):
            layers = [nn.Linear(in_dim, out_dim)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_dim, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, )
        )




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
