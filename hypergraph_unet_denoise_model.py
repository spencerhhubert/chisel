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
from blocks import DownBlock, UpBlock, AttnDownBlock, AttnUpBlock

class HypergraphUNet(nn.Module):
    def __init__(self):
        super().__init__()
        heads = 8
        projs = (16,32,48,64) #projections/block out channels
        time_embed_dim = 27 #intentionally random, does this matter?

        self.conv_in = HypergraphConv(3, projs[0])

        timestep_input_dim = projs[0]
        self.time_proj = Timesteps(first_inner_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([
            DownBlock(projs[0], projs[0]),
            AttnDownBlock(projs[0], projs[1], heads=heads, temb_channels=temb_channels),
            AttnDownBlock(projs[1], projs[2], heads=heads, temb_channels=temb_channels),
            AttnDownBlock(projs[2], projs[3], heads=heads, temb_channels=temb_channels),
        ])
        self.mid_block = AttnConvBlock(projs[3], projs[3], heads=heads, temb_channels=temb_channels)
        self.up_blocks = nn.ModuleList([
            AttnUpBlock(projs[3], projs[2], heads=heads, temb_channels=temb_channels),
            AttnUpBlock(projs[2], projs[1], heads=heads, temb_channels=temb_channels),
            AttnUpBlock(projs[1], projs[0], heads=heads, temb_channels=temb_channels),
            UpBlock(projs[0], projs[0]),
        ])
        #TODO: implement group norm out
        self.conv_out = HypergraphConv(projs[0], 3)

    def forward(self, x, incidence_matrix, timesteps):
        timesteps.to(x.device)
        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)
        x = self.conv_in(x, incidence_matrix)
        res_outs = (x,)
        for down_block in self.down_blocks:
            x, res_out = down_block(x, incidence_matrix, emb)
            res_outs += res_out #could be more than one
        x = self.mid_block(x, incidence_matrix, emb)
        for up_block, res_out in zip(self.up_blocks, reversed(res_outs)):
            x = up_block(x, incidence_matrix, emb, res_out)
        x = nn.SiLU(x)
        x = conv_out(x, incidence_matrix)
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
model = HypergraphUNet()

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
    for data_block in train_dataloader:
        data = [data_block[i:i+args["batch_size"]] for i in range(0, len(data_block), args["batch_size"])]
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

