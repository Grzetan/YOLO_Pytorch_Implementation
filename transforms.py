from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

"""
    All transforms except bboxes coordinates to be in YOLO format
    (midx, midy, w, h)
"""

class Resize():
    def __init__(self, w=416, h=416):
        self.w = w
        self.h = h
    
    def __call__(self, sample):
        image, bboxes_params = sample
        org_w, org_h = image.size
        image = image.resize((self.w, self.h))
        w_ratio = self.w / org_w
        h_ratio = self.h / org_h
        bboxes_params[...,0:1] *= w_ratio
        bboxes_params[...,1:2] *= h_ratio
        bboxes_params[...,2:3] *= w_ratio
        bboxes_params[...,3:4] *= h_ratio
        return image, bboxes_params

class Normalize():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.norm = T.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, bboxes_params = sample
        image = self.norm(image)
        return image, bboxes_params

class ToTensor():
    def __call__(self, sample):
        image, bboxes_params = sample
        w,h = image.size
        image = np.asarray(image)
        image = image / 255
        image = torch.tensor(image, dtype=torch.float)
        image = image.permute(2,0,1)
        bboxes_params[...,0:1] /= w
        bboxes_params[...,1:2] /= h
        bboxes_params[...,2:3] /= w
        bboxes_params[...,3:4] /= h
        bboxes_params = torch.tensor(bboxes_params, dtype=torch.float)
        return image, bboxes_params
