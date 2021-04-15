import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, shape, input_dim=64, output_size=32, output_channel=3):
        """
        :param shape: b, h, c, w
        :param input_dim: generator filters dim
        :param output_size:
        """
        super(Generator, self).__init__()
        image_size = output_size
        
