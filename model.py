import torch
from torch import nn
from torchvision import models


# use resnet18 for image encoding
class SingleImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights='DEFAULT')
        self.encoder = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(512, 256)

    def forward(self, x):
        """Image Encoding
            x: [B, C, H, W] -> [B, 256, 1]
        """
        x = self.encoder(x).view(x.shape[0], 512)
        x = self.fc(x).view(x.shape[0], 256, -1)
        return x


# adaptive batch normalization
class AdaBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels, affine=False)
        self.gamma = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.beta = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
    
    def forward(self, x, cond):
        """[B, Cin, N] -> [B, Cout, N] AdaBN(h, y) = ys BatchNorm(h) + yb"""
        h = self.norm(x)
        g = self.gamma(cond)
        b = self.beta(cond)
        out = h * g + b
        return out


# adaptive residual block
class AdaResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels):
        super().__init__()
        self.first_block = nn.ModuleDict(dict(
            norm = AdaBatchNorm(cond_channels, in_channels),
            relu = nn.ReLU(),
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False) 
        ))
        self.second_block = nn.ModuleDict(dict(
            norm = AdaBatchNorm(cond_channels, out_channels),
            relu = nn.ReLU(),
            conv = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        ))
    
    def forward(self, x, cond):
        """conditional residual block
            x: [B, Cin N] -> [B, Cout, N]
            cond: [B, 1000, 1]
        """
        # first block
        h = self.first_block.norm(x, cond)
        h = self.first_block.relu(h)
        h = self.first_block.conv(h)

        # second block
        h = self.second_block.norm(h, cond)
        h = self.second_block.relu(h)
        h = self.second_block.conv(h)

        # residual connection
        return x + h


# adaptive decoder
class AdaDecoder(nn.Module):
    def __init__(self, embed_param, cond_params, conv_param):
        super().__init__()
        self.embedder = nn.Conv1d(**embed_param)
        self.resblocks = nn.ModuleList([
            AdaResBlock(**cond_param) for cond_param in cond_params
        ])
        self.norm = AdaBatchNorm(cond_params[-1].cond_channels, cond_params[-1].out_channels)
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(**conv_param),
            nn.Sigmoid()
        )
    
    def forward(self, x, cond):
        """from position to occupancy
            x: [B, 3, N] -> [B, N]
            cond: [B, N]
        """
        assert len(x.shape) == 3
        B, _, N = x.shape
        x = self.embedder(x)
        for resblock in self.resblocks:
            x = resblock(x, cond)
        x = self.norm(x, cond)
        x = self.conv(x)
        x = x.view(B, N)
        return x
