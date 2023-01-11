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
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
import time
from functions import makeHyperIncidenceMatrix, makeHyperEdgeFeatures

class AttentionBlock(nn.Module):
    #hypergraph attention
    #perform attention on incidence matrix, designed to be used before conv and such
    #https://arxiv.org/abs/1901.08150
    def __init__(self):
        super().__init__()
    def forward(x, indidence_matrix):
        pass

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

class UpSampleBlock(nn.Module):
    def __init__(
        self,
        ratio=2
    ):
        super().__init__()

    def forward(self):
        return None

#simpler than attn block, attn block is just conv + attn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels) #TODO experiment with dropout

    def forward(self, x, incidence_matrix, temb):
        return self.conv(x, incidence_matrix)

class AttnConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, temb_channels=512, dropout=0.8):
        super().__init__()
        self.conv = HypergraphConv(in_channels, out_channels, heads=heads, use_attention=True, bias=True, concat=True, dropout=dropout) 
        time_emb_proj_out_channels = out_channels
        self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        self.nonlinearity = lambda x: F.silu(x) 
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, incidence_matrix, temb):
        hyperedge_features = makeHyperEdgeFeatures(batch.pos, batch.edge_index)
        hidden_state = self.conv(x, incidence_matrix, hyperedge_attr=hyperedge_features)
        temb = self.time_emb_proj(self.nonlinearity(temb))#[:, :, None, None] (not sure what this is about, makes incompatible shape)
        hidden_state += temb
        hidden_state = self.nonlinearity(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state

class UndoNoise(nn.Module):
    def __init__(self):
        super().__init__()
        inner_dim = 16
        time_embed_dim = inner_dim * 4 #should be *2 or *1?
        heads = 8

        self.encode = AttnConvBlock(3, inner_dim, temb_channels=time_embed_dim, heads=heads)
        self.decode = AttnConvBlock(inner_dim, 3, heads=1, temb_channels=time_embed_dim)

        #fourier projection causes nan
        #self.time_proj = GaussianFourierProjection(embedding_size=inner_dim, scale=16)
        #timestep_input_dim = inner_dim * 2
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        timestep_input_dim = inner_dim

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

    def forward(self, x, incidence_matrix, timesteps):
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        x = self.encode(x, incidence_matrix, emb)
        x = self.decode(x, incidence_matrix, emb)
        return x

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
    "device": "cpu",
    "batch_size": 1,
    "data_path": "/Volumes/PortableSSD/data/ABC-Dataset",
}

from ABCDataset import ABCDataset2
train_dataloader = ABCDataset2(args["data_path"]) #stored in blocks of 1024
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

model.train()
for epoch in range(args["num_epochs"]):
    for block in train_dataloader:
        data = [block[i:i+args["batch_size"]] for i in range(0, len(block), args["batch_size"])]
        for step,batch in enumerate(train_dataloader):
            batch = batch[0]#[0] is because only doing batch size of 1 for now because not properly batched in data set processing
            batch.edge_index = makeHyperIncidenceMatrix(batch)
            batch.edge_attr = makeHyperEdgeFeatures(batch.pos, batch.edge_index)
            batch.to(args["device"])
            clean_verts = batch.pos
            noise = torch.randn(clean_verts.shape).to(clean_verts.device)
            bsz = clean_verts.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_verts.device
            ).long()

            #timesteps = torch.randint(0,1,(bsz,),device=clean_verts.device) #for testing, train on a single noise step

            noisy_verts = noise_scheduler.add_noise(clean_verts, noise, timesteps)

            batch.edge_index = makeHyperIncidenceMatrix(batch)
            model_output = model(noisy_verts, batch.edge_index, timesteps) #problem: mean of output is infinity

            #assume epsilon prediction
            print("got model output")
            loss = F.mse_loss(model_output, noise) #need to come back and modify, use pytorch3d losses to account for like flatness of surfaces and stuff

            print(f"loss: {loss}")

            loss.backward()
            print("made it past backwards")
            exit()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            print(f"completed step {step} in epoch {epoch}")

        if epoch % args["save_epochs"] == 0 or epoch == args["num_epochs"] - 1:
            #torch.save(model, f"models/{time.time()}_epoch{epoch}.pt") #doesn't work rn idk why
            pass

#gatconv has edge update, which acts like concat in regular attn
#hyperedges are preprented by the edge_idx and another array of the same size saying what index each edge is

