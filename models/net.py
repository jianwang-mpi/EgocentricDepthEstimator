from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
import models.modules as modules
from torchvision import utils

import models.senet as senet
import models.resnet as resnet
import models.densenet as densenet

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, deformable=False):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel, deformable=deformable)
        self.R = modules.R(block_channel)


    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out
