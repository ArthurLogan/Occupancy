import torch
from torch import nn


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
        """[B, Cin N] -> [B, Cout, N]"""

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
        self.conv = nn.Sequential(
            AdaBatchNorm(cond_params[-1].cond_channels, cond_params[-1].out_channels),
            nn.ReLU(),
            nn.Conv1d(**conv_param)
        )
    
    def forward(self, x, cond):
        """[B, 3, N] -> [B, 1, N] from position to occupancy"""
        x = self.embedder(x)
        for resblock in self.resblocks:
            x = resblock(x, cond)
        x = self.conv(x)
        return x
