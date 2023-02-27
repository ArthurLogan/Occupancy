import torch
from torch import nn
from torch import optim
from torch import utils
from torchvision import models

from tensorboardX import SummaryWriter

from model import AdaDecoder
from dataset import ShapeNet


# convert dict to var
class Dict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


def train(config):
    # device
    device = torch.device(config["device"])



def eval(config):
    pass
