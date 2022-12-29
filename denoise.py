import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv, fps, TopKPooling
import torch_geometric.transforms as T
import os
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
import time

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

    def forward(self, x, hyperedge_index, timesteps):
        x = self.encode(x, hyperedge_index)
        x = self.decode(x, hyperedge_index)
        return x

#Mesh -> Data (mesh -> hypergraph)
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

def makeFaceTensor(incidenceMat):
    #as is the edges never even change so this isn't necessary but if the model gets more complex to account for clipping edges then perhaps it'll modify edges and then we'll need this
    return None

def applyNoise(x,distribution):
    return x + (torch.randn(x.shape) * (distribution ** 0.5))

from ABCDataset import ABCDataset2
train_dataloader = ABCDataset2("/Volumes/PortableSSD/data/ABC-Dataset")

args = {
    "learning_rate": 1e-4,
    "adam_beta1": 0.95,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-6,
    "adam_epsilon": 1e-08,
    "lr_scheduler": "cosine",
    "lr_warmup_steps": 500,
    "num_epochs": 100,
    "gradient_accumulation_steps": 1,
    "ddpm_num_steps": 1000,
    "ddpm_beta_schedule": "linear",
    "prediction_type": "epsilon",
    "save_epochs": 2,
}

model = UndoNoise()

noise_scheduler = DDPMScheduler(
    num_train_timesteps=args["ddpm_num_steps"],
    beta_schedule=args["ddpm_beta_schedule"],
    prediction_type=args["prediction_type"],
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args["learning_rate"],
    betas=(args["adam_beta1"], args["adam_beta2"]),
    weight_decay=args["adam_weight_decay"],
    eps=args["adam_epsilon"],
)

lr_scheduler = get_scheduler(
    args["lr_scheduler"],
    optimizer=optimizer,
    num_warmup_steps=args["lr_warmup_steps"],
    num_training_steps=(len(train_dataloader) * args["num_epochs"]) // args["gradient_accumulation_steps"],
)

for epoch in range(args["num_epochs"]):
    model.train()
    for step,batch in enumerate(train_dataloader):
        clean_verts = batch.pos
        noise = torch.randn(clean_verts.shape).to(clean_verts.device)
        bsz = clean_verts.shape[0]
        #timesteps = torch.randint(
        #    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_verts.device
        #).long()

        timesteps = torch.randint(0,1,(bsz,),device=clean_verts.device) #just for testing

        noisy_verts = noise_scheduler.add_noise(clean_verts, noise, timesteps)

        batch.edge_index = makeHyperIncidenceMatrix(batch)
        model_output = model(noisy_verts, batch.edge_index, timesteps)

        #assume epsilon prediction
        loss = F.mse_loss(model_output, noise) #need to come back and modify, use pytorch3d losses to account for like flatness of surfaces and stuff

        print(loss)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    if epoch % args["save_epochs"] == 0 or epoch == args["num_epochs"] - 1:
        torch.save(model, f"models/{time.time()}_epoch{epoch}.pt")
                



#gatconv has edge update, which acts like concat in regular attn

#hyperedges are preprented by the edge_idx and another array of the same size saying what index each edge is


#----
exit()
from ABCDataset import ABCDataset2
data = ABCDataset2("/Volumes/PortableSSD/data/ABC-Dataset")
mesh = data[0][1]

save_obj("original.obj", mesh.pos, mesh.face.t())

mesh.edge_index = makeHyperIncidenceMatrix(mesh)
mesh.pos = applyNoise(mesh.pos,1) #faces will clip quickly here. not good.
save_obj("noisy.obj", mesh.pos, mesh.face.t())

net = UndoNoise()
y = net(mesh.pos, mesh.edge_index)
mesh.pos = y

save_obj("model_output.obj", mesh.pos, mesh.face.t())


