import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from utils import plot_sample, iou

class COCO2017(Dataset):
    def __init__(self, dataset_path, annotation_path, images_path, anchors, n_classes, SCALES=[13,26,53], transform=None):
        self.dataset_path = dataset_path
        f = open(os.path.join(dataset_path, annotation_path), 'r')
        self.annotations = f.readlines()
        f.close()
        self.images_path = os.path.join(dataset_path, images_path)
        self.transform = transform
        self.anchors = torch.tensor(anchors)
        self.anchors_per_scale = len(self.anchors) // len(SCALES)
        self.n_classes = n_classes
        self.SCALES = SCALES
        self.instances_per_scale = [S * S for S in self.SCALES]


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx].split(' ')
        img_path = None
        # bboxes_params have shape (n, 5) where n in number of objects on image
        # and 5 means (x1, y1, x2, y2, class_label) 
        bboxes_params = np.zeros((len(data) - 1, 5), dtype=np.int16)

        for i, elem in enumerate(data):
            # Image path is first
            if i == 0:
                img_path = elem.strip()
                continue
            
            corr = elem.split(',')

            bboxes_params[i-1,:] = [int(n) for n in corr]

        img = Image.open(os.path.join(self.images_path, img_path)).convert('RGB')
        img = np.asarray(img)

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes = bboxes_params)
            img = transformed['image']
            bboxes_params = transformed['bboxes']
        bboxes_params = torch.tensor(bboxes_params)

        # Build targets
        n_bboxes = sum([self.anchors_per_scale * S for S in self.instances_per_scale]) # Total number of bboxes across all scales
        targets = torch.zeros((n_bboxes, 6)) # For every bbox there is x, y, w, h, objectness score and class label

        for bbox in bboxes_params:
            # Calculate IOU with every anchor
            ious = iou(bbox[...,2:4].unsqueeze(0), self.anchors, only_size=True)


        return img, bboxes_params

if __name__ == '__main__':
    import config
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import albumentations as A

    f = open(os.path.join(config.DATASET_PATH, 
                          config.TRAIN_PATH,
                          config.CLASSES_PATH), 'r')
    class_names = [cls.strip() for cls in f.readlines()]

    transform = A.Compose([
        A.Resize(width=450, height=450),
        A.HorizontalFlip(p=0.5)

    ], bbox_params=A.BboxParams(format='pascal_voc'))

    dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                       config.ANNOTATIONS_PATH, 
                       config.IMAGES_PATH,
                       config.ANCHORS,
                       config.N_CLASSES,
                       transform=transform)

    for i in range(100, 110):
        img, bboxes_params = dataset[i]
        plot_sample(img, bboxes_params, class_names)

