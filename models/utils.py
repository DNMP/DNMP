import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    '''
    Position Encoding for xyz/directions
    Args:
        in_channels: number of input channels (typically 3)
        N_freqs: maximum frequency
        logscale: if True, use log scale for frequencies
    Inputs:
        x: (batch_size, in_channels)
    '''
    def __init__(self, in_channels, N_freqs, logscale=True):
        super(PositionEncoding, self).__init__()

        self.in_channels = in_channels
        self.N_freqs = N_freqs

        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    
    def forward(self, x):
        
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(x * freq))
        
        out = torch.cat(out, -1)
        return out

class MLPs(nn.Module):
    '''
    Args:
        net_depth: number of layers in the network
        net_width: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz
        in_channels_dir: number of input channels for directions
        skips: list of integers indicating which layers to skip
        use_viewdirs: if True, use view directions as an input to infer RGB
    Inputs:
        x: (batch_size, num_samples, channels) [xyz, directions]
    '''
    def __init__(self, 
                 net_depth=8, 
                 net_width=256,
                 in_channels=32,
                 skips=[4]):
        super(MLPs, self).__init__()

        self.net_depth = net_depth
        self.net_width = net_width
        self.skips = skips

        self.in_channels = in_channels

        self.layers = nn.ModuleList()

        for i in range(self.net_depth):
            if i == 0:
                layer = nn.Linear(in_channels, self.net_width)
            elif i in skips:
                layer = nn.Linear(self.net_width+in_channels, self.net_width)
            else:
                layer = nn.Linear(self.net_width, self.net_width)
            self.layers.append(layer)

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i in self.skips:
                h = torch.cat([h, x], -1)
            h = F.relu(layer(h))
        
        return h

class BackgroundMLP(nn.Module):

    def __init__(self,
                 net_depth=2,
                 net_width=256,
                 in_channels_dir=27,
                 skips=[]):
        super(BackgroundMLP, self).__init__()

        self.net_depth = net_depth
        self.net_width = net_width
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        self.viewdir_layers = nn.ModuleList()
        for i in range(self.net_depth):
            if i == 0:
                layer = nn.Linear(in_channels_dir, self.net_width)
            elif i in skips:
                layer = nn.Linear(self.net_width+in_channels_dir, self.net_width)
            else:
                layer = nn.Linear(self.net_width, self.net_width)
            self.viewdir_layers.append(layer)
        
        self.rgb_layer = nn.Linear(self.net_width, 3)
    
    def forward(self, x):

        h = x
        for i, layer in enumerate(self.viewdir_layers):
            if i in self.skips:
                h = torch.cat([h, x], -1)
            h = F.relu(layer(h))
        
        rgb = torch.sigmoid(self.rgb_layer(h))
        
        return rgb