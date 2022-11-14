import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, Dataset
from pytorch3d.io import load_objs_as_meshes
from torch_geometric.transforms import face_to_edge

#assume path like "data/ABC-Dataset/abc_0000_obj_v00/some.obj"

class ABCDataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_file_names(self) -> List[str]:
        return ["abc_0000_obj_v00"]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_0.pt', 'data_1.pt', 'data_10.pt', 'data_11.pt', 'data_12.pt', 'data_13.pt', 'data_14.pt', 'data_15.pt', 'data_16.pt', 'data_17.pt', 'data_18.pt', 'data_19.pt', 'data_2.pt', 'data_20.pt', 'data_21.pt', 'data_22.pt', 'data_23.pt', 'data_24.pt', 'data_25.pt', 'data_26.pt', 'data_27.pt', 'data_28.pt', 'data_3.pt', 'data_4.pt', 'data_5.pt', 'data_6.pt', 'data_7.pt', 'data_8.pt', 'data_9.pt']


    def process(self):
        folders = os.listdir(self.root)
        folders = [x for x in folders if "obj" in x]
        idx = 0
        for folder in folders:
            path = os.path.join(self.root, folder)
            objs = os.listdir(path)
            objs = [os.path.join(path,x) for x in objs if ".obj" in x]
            num_objs_per_datablock=16
            for i in range(0,len(objs),num_objs_per_datablock):
                meshes = load_objs_as_meshes(objs[i:i+num_objs_per_datablock])
                data_list = []
                for mesh in meshes:
                    faces = mesh.faces_packed()
                    verts = mesh.verts_packed()
                    edges = mesh.edges_packed().t().contiguous()
                    data_list.append(Data(x=verts, edge_index=edges))
                torch.save(data_list, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx+=1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
