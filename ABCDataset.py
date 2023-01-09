import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.io import read_obj
#from torch_geometric.nn import radius, fps, knn, ball_query
from pytorch3d.io import load_objs_as_meshes

#assume path like "data/ABC-Dataset/abc_0000_obj_v00/some.obj"
class ABCDataset2(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_file_names(self) -> List[str]:
        return ["abc_0000_obj_v00"]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_0.pt', 'data_1.pt', 'data_2.pt', 'data_3.pt']

    def download(self):
        pass

    def process(self):
        folders = list(filter(lambda x : "obj" in x and not ".7z" in x, os.listdir(self.root))) #get all the folders that aren't compressed and contain objs
        idx = 0
        for folder in folders:
            path = os.path.join(self.root, folder)
            objs = [os.path.join(path,x) for x in os.listdir(path) if ".obj" in x] #get list of all .obj files in the folder

            #appears the 7z decompression schemes are different on linux and mac so we account for both scenarios. the .objs end up in folders on mac
            more_folders = filter(os.path.isdir, map(lambda x : os.path.join(path,x),os.listdir(path)))
            more_folders = map(os.path.basename, more_folders)
            more_folders = list(more_folders)
            for another_folder in more_folders:
                path = os.path.join(self.root, folder, another_folder)
                objs += [os.path.join(path,x) for x in os.listdir(path) if ".obj" in x and not "._" in x]

            num_objs_per_datablock=1024 #like 17k items in dir

            for i in range(0,len(objs),num_objs_per_datablock):
                meshes = []
                for obj in objs[i:i+num_objs_per_datablock]:
                    meshes.append(read_obj(obj))
                data_list = []
                for mesh in meshes:
                    #worth returning to these ideas:
                    #pairwise_dist = pairwise_distance(verts, faces)
                    #seed_index = fps(verts, batch_size=1024)
                    #edge_index = knn(pairwise_dist, k=10, batch_size=1024)
                    #face_index = ball_query(verts, verts[seed_index], radius, batch_size=1024)

                    data_list.append(mesh)
                    #could add code to construct proper edges where the weight value is the actual distance between the points

                #batch = Batch.from_data_list(data_list) #makes mega graph where all the actual graphs are present but disconnected. might return to
                torch.save(data_list, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx+=1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))

if __name__ == "__main__":
    data = ABCDataset2("data/ABC-Dataset/") #do processing automatically if we call this class. takes a long time but gets saved to hdd
    data.process()
