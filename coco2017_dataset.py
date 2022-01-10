import torch 
import torch.nn as nn
from torch.optim.data import Dataset
import os
from PIL import Image

class COCO2017(Dataset):
    def __init__(self, dataset_path, annotation_path, images_path, transform=None):
        self.dataset_path = dataset_path
        f = open(annotation_path, 'r')
        self.annotations = f.readlines()
        f.close()
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, )