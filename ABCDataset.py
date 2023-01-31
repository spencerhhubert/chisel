import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.io import read_obj
#from torch_geometric.nn import radius, fps, knn, ball_query
from pytorch3d.io import load_objs_as_meshes

#assume path like "data/ABC-Dataset/abc_0000_obj_v00/some.obj"
class ABCDataset2(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, pre_transform=None, pre_filter=None, batched:bool=True):
        self.batched = batched
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_file_names(self) -> List[str]:
        return ["abc_0000_obj_v00"]

    @property
    def processed_file_names(self) -> List[str]:
        if self.batched:
            return ['data_batched_0.pt', 'data_batched_1.pt', 'data_batched_10.pt', 'data_batched_11.pt', 'data_batched_12.pt', 'data_batched_13.pt', 'data_batched_14.pt', 'data_batched_15.pt', 'data_batched_16.pt', 'data_batched_17.pt', 'data_batched_18.pt', 'data_batched_2.pt', 'data_batched_3.pt', 'data_batched_4.pt', 'data_batched_5.pt', 'data_batched_6.pt', 'data_batched_7.pt', 'data_batched_8.pt', 'data_batched_9.pt']
        else:
            return ['data_26.pt', 'data_44.pt', 'data_0.pt', 'data_1.pt', 'data_10.pt', 'data_11.pt', 'data_12.pt', 'data_13.pt', 'data_14.pt', 'data_15.pt', 'data_16.pt', 'data_17.pt', 'data_18.pt', 'data_19.pt', 'data_2.pt', 'data_20.pt', 'data_21.pt', 'data_22.pt', 'data_23.pt', 'data_24.pt', 'data_25.pt', 'data_27.pt', 'data_28.pt', 'data_29.pt', 'data_3.pt', 'data_30.pt', 'data_31.pt', 'data_32.pt', 'data_33.pt', 'data_34.pt', 'data_35.pt', 'data_36.pt', 'data_37.pt', 'data_38.pt', 'data_39.pt', 'data_4.pt', 'data_40.pt', 'data_41.pt', 'data_42.pt', 'data_43.pt', 'data_45.pt', 'data_46.pt', 'data_47.pt', 'data_48.pt', 'data_49.pt', 'data_5.pt', 'data_50.pt', 'data_51.pt', 'data_52.pt', 'data_53.pt', 'data_54.pt', 'data_55.pt', 'data_56.pt', 'data_57.pt', 'data_58.pt', 'data_59.pt', 'data_6.pt', 'data_60.pt', 'data_61.pt', 'data_62.pt', 'data_7.pt', 'data_8.pt', 'data_9.pt']

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
                if self.batched:
                    batch = Batch.from_data_list(data_list) #makes mega graph where all the actual graphs are present but disconnected
                    torch.save(batch, os.path.join(self.processed_dir, f"data_batched_{idx}.pt"))
                else:
                    torch.save(data_list, os.path.join(self.processed_dir, f"data_{idx}.pt"))
                idx+=1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.batched:
            return torch.load(os.path.join(self.processed_dir, f"data_batched_{idx}.pt"))
        else:
            return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))

if __name__ == "__main__":
    data = ABCDataset2("data/ABC-Dataset/", batched=True) #do processing automatically if we call this class. takes a long time but gets saved to hdd
    data.process()
