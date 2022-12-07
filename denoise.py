import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv, fps, TopKPooling
import torch_geometric.transforms as T
import os
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes

class DownSampleBlock(nn.Module):
    def __init__(
        self,
        ratio=0.5
    ):
        super().__init__()
        self.downsampler = TopKPooling(3)

    def forward(self, data):
#        x, edge_index, _, _ = fps(data.x, data.edge_index)
        return self.downsampler(data.x, data.edge_index)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels) #TODO experiment with dropout

    def forward(self, x, hyperedge_index):
        return self.conv(x, hyperedge_index)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8):
        super().__init__()
        #how make bias work with true
        self.conv = HypergraphConv(in_channels, out_channels, heads=heads, bias=False , dropout=0.5) #TODO experiment with dropout

    def forward(self, x, hyperedge_index):
        return self.conv(x, hyperedge_index)

class UndoNoise(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = ConvBlock(3,16)
        self.decode = ConvBlock(16,3)

    def forward(self, x, hyperedge_index):
        x = self.encode(x, hyperedge_index)
        x = self.decode(x, hyperedge_index)
        return x

def makeHyperIncidenceMatrix(mesh):
    faces = mesh.face.t()
    for i,face in enumerate(faces):
        idx = torch.tensor([i]*len(face))
        hyperedge = torch.stack((face,idx),dim=0)
        if not 'out' in locals():
            out = hyperedge
        else:
            out = torch.cat((out,hyperedge),dim=-1)
    return out

def applyNoise(x,distribution):
    return torch.randn(x.shape) * (distribution ** 0.5)

from ABCDataset import ABCDataset2
mesh = ABCDataset2("data/ABC-Dataset")[0][0]
mesh.edge_index = makeHyperIncidenceMatrix(mesh)
noisy_mesh = mesh
noisy_mesh.pos = applyNoise(mesh.pos,1000)

net = UndoNoise()
y = net(noisy_mesh.pos, noisy_mesh.edge_index)
new_mesh = noisy_mesh
new_mesh.pos = y
print(y.shape)



#gatconv has edge update, which acts like concat in regular attn

#hyperedges are preprented by the edge_idx and another array of the same size saying what index each edge is

#final_obj = os.path.join('./', 'final_model.obj')
#save_obj(final_obj, x, final_faces)
