import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, Linear
from functions import makeHyperEdgeFeatures

class DownSampleBlock(nn.Module):
    def __init__(self, ratio=0.5):
        super().__init__()
        self.downsampler = TopKPooling(3)

    def forward(self, data):
#        x, edge_index, _, _ = fps(data.x, data.edge_index)
        return self.downsampler(data.x, data.edge_index)

class UpSampleBlock(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()

    def forward(self):
        return None

#Conv block with optional attention
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.5, temb_channels=512, use_attention:bool=False):
        super().__init__()
        self.use_attention = use_attention
        self.conv = HypergraphConv(in_channels, out_channels, heads=heads, use_attention=use_attention, bias=True, concat=use_attention, dropout=dropout) 
        self.proj_down = Linear(out_channels*heads, out_channels)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels) #mimic resnet
        self.nonlinearity = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, incidence_matrix, temb, res_in=None):
        hyperedge_features = None
        if self.use_attention:
            hyperedge_features = makeHyperEdgeFeatures(x, incidence_matrix)
        hidden_state = self.conv(x, incidence_matrix, hyperedge_attr=hyperedge_features)
        hidden_state = self.proj_down(hidden_state)
        #should be norm here
        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)#[:, :, None, None] 
        hidden_state += temb
        hidden_state = self.nonlinearity(hidden_state)
        hidden_state = self.dropout(hidden_state)
        #TODO: add risidual output for cross connection
        res_out = (torch.zeros_like(x),) if res_in is None else None
        return (hidden_state, res_out)

class AttnDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.5, temb_channels=512):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, heads, dropout, use_attention=True, temb_channels=temb_channels)

    def forward(self, x, incidence_matrix, temb):
        return self.conv(x, incidence_matrix, temb)

class AttnUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.5, temb_channels=512):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, heads, dropout, use_attention=True, temb_channels=temb_channels)

    def forward(self, x, incidence_matrix, temb, res_in):
        return self.conv(x, incidence_matrix, temb, res_in)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=512):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, temb_channels=temb_channels)

    def forward(self, x, incidence_matrix, temb):
        return self.conv(x, incidence_matrix, temb)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=512):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, temb_channels=temb_channels)

    def forward(self, x, incidence_matrix, temb, res_in):
        return self.conv(x, incidence_matrix, temb, res_in)
