import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from ABCDataset import ABCDataset

dataset = ABCDataset("data/ABC-Dataset")
data = dataset[0][0]
data2 = dataset[0][1]

class CapyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pool_ratios = [2000 / data.num_nodes, 0.5]
        self.unet = GraphUNet(in_channels=3, hidden_channels=32,
                              out_channels=3, depth=4,
                              pool_ratios=pool_ratios)

    def forward(self, data):
        x = self.unet(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)

net = CapyNet()
y = net.forward(data)
y2 = net.forward(data2)
print(data.x.shape)
print(y.shape)
print(y2.shape)

def printAsciiBananaPhone(width):
    for i in range(width):
        print(" " * (width - i), end="")
        for j in range(2 * i + 1):
            if i == 0 or i == width - 1:
                print("0", end="")
            elif i == width - 2:
                print("0", end="")
            else:
                if j == 0 or j == 2 * i:
                    print("0", end="")
                else:
                    print(" ", end="")
        print()
