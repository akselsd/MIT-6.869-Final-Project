import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, models, transforms
import os   
from PIL import Image
from datetime import datetime

class GazeboDataset(Dataset):

    # NB NB: All indexes in poses.txt is increased by 1 as compared to the corresponding image file
    # But we are using 0 index for the list, so its ok
    def __init__(self, fn, img_transform, do_preloading):
        self.root = fn
        self.img_transform = img_transform
        self.target_transform = None
        self.size = len(os.listdir(os.path.join(self.root, "images/")))

        with open(os.path.join(self.root, "images_pose/poses.txt"), "r") as f:
            poses = f.readlines()
        
        self.pose = [i.strip().split()[1:4] for i in poses]
        self.pose = [[float(i) for i in j] for j in self.pose]
        self.pose = torch.Tensor(self.pose)
        
        
        if len(self.pose) != self.size:
            print(self.size)
            print(len(self.pose))
            raise Exception("Number of poses is not equal number of images")
        

        self.preloaded = False
        
        if do_preloading:
            print("Preloading images")
            self.images = []
            for idx in range(self.size):
                self.images.append(self.pil_loader(idx))
                if not idx % 50:
                    print(idx, end="\r", flush=True)
            
            print("\nPreloading done")
            self.preloaded = True
        else:
            print("Skipping preloading")



    def pil_loader(self, idx):
        if self.preloaded:
            return self.images[idx]
        
        path = self.get_path(idx)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return self.img_transform(img.convert('RGB'))
    
    def get_path(self, idx):
        path = "images/img" + str(idx) + ".jpeg"
        return os.path.join(self.root, path)

    # Overloaded functions needed by torch.utils.data.DataLoader
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        
        sample = self.pil_loader(idx)
        target = self.pose[idx]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def get_dataloaders(dataset, batch_size):
    ratios = [0.7, 0.2, 0.1]
    sizes = [int(len(dataset)*i) for i in ratios]
    sizes[-1] = len(dataset) - sum(sizes[:-1])
    datasets = torch.utils.data.random_split(dataset, sizes)
    return [DataLoader(i, batch_size = batch_size, shuffle = True) for i in datasets]
