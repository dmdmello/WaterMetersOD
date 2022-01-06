#python imports
import ast
import re
import shutil
import os
import math
import copy

#torch imports
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

#other imports
import pandas 
import cv2
import matplotlib.pyplot as plt
import numpy as np

class TlkWaterMetersDataset(VisionDataset):
    #custom pytorch dataset for TlkWaterMeters
    
    def __init__(self,df,img_folder,transform):
        super().__init__(img_folder, transform=transform, target_transform=None)
        self.df = df
        self.transform = transform
        self.img_folder = img_folder
        self.image_names = self.df[:]['photo_name']
        self.labels = list(self.df[:]['bb_coordinates'])
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self,index):
        img_path = os.path.join(self.img_folder, self.image_names.iloc[index])
        if not os.path.isfile(img_path):
            raise Exception("Image path {} does not exist".format(img_path)) 
        image=cv2.imread(img_path)
        if image is not None:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=self.transform(image)
            targets=self.labels[index]
            sample = image, torch.Tensor(targets)
            return sample
        else:
            raise Exception("Error reading image") 

def create_dl(df, img_folder, batch_size=64, use_color_aug=True, drop_last=True, shuffle=True):
    if use_color_aug:
        T = transforms.Compose([transforms.ToTensor(), 
                                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4), 
                                transforms.Normalize([.5],[.5],[.5])])
    else:
        T = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize([.5],[.5],[.5])])      
    tlk_wk_dataset =TlkWaterMetersDataset(df=df,img_folder=img_folder, transform=T)
    dl = torch.utils.data.DataLoader(tlk_wk_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, sampler=None)   
    return dl
  
