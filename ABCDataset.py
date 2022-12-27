import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.io import read_obj
#from torch_geometric.nn import radius, fps, knn, ball_query
from pytorch3d.io import load_objs_as_meshes

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
        return ["data_0.pt"]

    def process(self):
        folders = os.listdir(self.root)
        folders = [x for x in folders if "obj" in x]
        idx = 0
        for folder in folders:
            path = os.path.join(self.root, folder)
            objs = os.listdir(path)
            objs = [os.path.join(path,x) for x in objs if ".obj" in x]
            num_objs_per_datablock=1024 #like 17k items in dir. so this will make ~17 processed items
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

class ABCDataset2(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_file_names(self) -> List[str]:
        return ["abc_0000_obj_v00"]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_0.pt', 'data_1.pt', 'data_2.pt', 'data_3.pt', 'data_4.pt', 'data_5.pt', 'data_6.pt', 'data_7.pt']

    def download(self):
        pass

    def process(self):
        folders = [x for x in os.listdir(self.root) if "obj" in x] #get all the folders in the root with "obj" in their name
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

            num_objs_per_datablock=2 #like 17k items in dir. so this will make ~17 processed items

            for i in range(0,len(objs),num_objs_per_datablock):
                meshes = []
                for obj in objs[i:i+num_objs_per_datablock]:
                    meshes.append(read_obj(obj)) #here it blows up

                data_list = []
                for mesh in meshes:
                    

                    #verts = mesh.verts_packed()
                    #faces = mesh.faces_packed()
                    #edges = faces.t().contiguous()

                    #pairwise_dist = pairwise_distance(verts, faces)
                    #seed_index = fps(verts, batch_size=1024)
                    #edge_index = knn(pairwise_dist, k=10, batch_size=1024)
                    #face_index = ball_query(verts, verts[seed_index], radius, batch_size=1024)

                    data_list.append(mesh)
                    #construct edges
                    #give edge attr of distance between points
                    #what was that reconstructing faces from sample points nonsense?

                torch.save(data_list, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx+=1
                #manual hard stop
                if idx == 8:
                    print("processed 8 mesh batches")
                    exit()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))


            
if __name__ == "__main__":
    ABCDataset2("data/ABC-Dataset/") #do processing automatically if we call this class. takes a long time but gets saved to hdd

