import os
import torch
from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
#breaks because I currently dont have the right cuda versions
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

data_path = "data/ABC-Dataset/abc_0000_obj_v00"

def getRanObj(path):
    choices = os.listdir(path)
    choices = [x for x in choices if not os.path.isdir(os.path.join(path,x))]
    from random import choice
    return os.path.join(path, choice(choices))

def getRanListObj(path):
    choices = os.listdir(path)
    choices = [x for x in choices if not os.path.isdir(os.path.join(path,x))]
    from random import sample
    objs = sample(choices, 2)
    objs = list(map(lambda x : os.path.join(path,x),objs))
    return objs

meshes = load_objs_as_meshes(getRanListObj(data_path))
packed = meshes.verts_packed()
padded = meshes.verts_padded() #this is how we get a batch 
print(padded)
exit()

#---
objs = getRanListObj("data/ABC-Dataset/abc_0000_obj_v00")
meshes = load_objs_as_meshes(objs)
datas = []
for mesh in meshes:
    verts = mesh.verts_packed()
    edges = mesh.edges_packed().t().contiguous()
    datas.append(Data(x=verts,edge_index=edges))

loader = DataLoader(datas,batch_size=2)
for batch in loader:
    print(batch)
exit()
#---


#---
target_obj = getRanObj(data_path)

verts, faces, aux = load_obj(target_obj)

faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)



exit()

n_faces = 128
mesh = Meshes(verts=[verts], faces=[faces_idx])
mesh = sample_points_from_meshes(mesh,n_faces)
print(mesh.shape)


exit()

#scale the target mesh to be inside a unit sphere
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

#mesh for target object
target_mesh = Meshes(verts=[verts], faces=[faces_idx])

#unit sphere to fit
src_mesh = ico_sphere(4, device)

deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

print(src_mesh.verts_packed())
print(src_mesh.verts_packed().shape)

optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

# Number of optimization steps
epochs = 2000

#we take the loss w/ respect to various parts of the obj
#we weight the loss produced by each part for the overall loss
#these should be learned weights too bruh
w_chamfer = 1.0
w_edge = 1.0
#mesh normal consistency
w_normal = 0.01
#mesh laplacian smoothing
w_laplacian = 0.1

n_points_to_sample = 5000

loop = tqdm(range(epochs))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    sample_trg = sample_points_from_meshes(target_mesh, n_points_to_sample)
    sample_src = sample_points_from_meshes(new_src_mesh, n_points_to_sample)

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
    loss_edge = mesh_edge_loss(new_src_mesh)

    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

    #note loss is a weighted sum
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

    #print loss
    loop.set_description('total_loss = %.6f' % loss)

    #save loss
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))

    loss.backward()
    optimizer.step()

# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = final_verts * scale + center

# Store the predicted mesh using save_obj
final_obj = os.path.join('./', 'final_model.obj')
save_obj(final_obj, final_verts, final_faces)
