import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from utils import plot_sample, iou, grid_to_linear, linear_to_grid

class COCO2017(Dataset):
    def __init__(self, dataset_path, annotation_path, images_path, anchors, IMG_SIZE=416, SCALES=[13,26,52], ignore_iou_thresh=0.5, transform=None):
        self.dataset_path = dataset_path
        f = open(os.path.join(dataset_path, annotation_path), 'r')
        self.annotations = f.readlines()
        f.close()
        self.images_path = os.path.join(dataset_path, images_path)
        self.transform = transform
        self.anchors = torch.tensor(anchors) / IMG_SIZE
        self.anchors_per_scale = len(self.anchors) // len(SCALES)
        self.SCALES = SCALES
        self.IMG_SIZE = IMG_SIZE
        self.ignore_iou_thresh = ignore_iou_thresh
        self.instances_per_scale = [S * S for S in self.SCALES]

    def get_loader(self, batch_size):
        loader = DataLoader(self, batch_size=batch_size, drop_last=True, shuffle=True)
        return loader

    def __len__(self):
        return 100
        # return len(self.annotations)

    def __getitem__(self, idx):
        data = self.annotations[idx].split(' ')
        img_path = None
        # bboxes_params have shape (n, 5) where n in number of objects on image
        # and 5 means (x1, y1, x2, y2, class_label) 
        bboxes_params = np.zeros((len(data) - 1, 5))

        for i, elem in enumerate(data):
            # Image path is first
            if i == 0:
                img_path = elem.strip()
                continue
            
            corr = elem.split(',')

            bboxes_params[i-1,:] = [int(n) for n in corr]

        # Read image
        img = Image.open(os.path.join(self.images_path, img_path)).convert('RGB')
        w, h = img.size
        # Ensure that all bboxes are inside of the image
        bboxes_params[...,2:3][bboxes_params[...,2:3] > w] = w
        bboxes_params[...,3:4][bboxes_params[...,3:4] > h] = h
        # Convert to x, y, w, h
        bboxes_params[...,2:3] = bboxes_params[...,2:3] - bboxes_params[...,0:1]
        bboxes_params[...,3:4] = bboxes_params[...,3:4] - bboxes_params[...,1:2]
        # Convert to midx, midy, w, h
        bboxes_params[...,0:1] = bboxes_params[...,0:1] + bboxes_params[...,2:3]/2
        bboxes_params[...,1:2] = bboxes_params[...,1:2] + bboxes_params[...,3:4]/2
        # Ensure that all bboxes have width and height of at least 1
        bboxes_params[...,2:4][bboxes_params[...,2:4] == 0] = 1

        if self.transform is not None:
            img, bboxes_params = self.transform((img, bboxes_params))

        if not isinstance(bboxes_params, torch.Tensor):
            bboxes_params = torch.tensor(bboxes_params)

        # Build targets
        n_bboxes = sum([self.anchors_per_scale * S * S for S in self.SCALES]) # Total number of bboxes
        bboxes_params[...,0:4] /= self.IMG_SIZE

        targets = torch.zeros((n_bboxes, 6)) # x, y, w, h, objetness, class_label

        for k, bbox in enumerate(bboxes_params):
            # Calculate IOU with every anchor
            ious = iou(bbox[...,2:4], self.anchors, only_size=True).squeeze()
            ious_args = torch.argsort(ious, dim=0, descending=True)
            anchor_found = False

            for anchor in ious_args:
                if anchor_found and ious[anchor] < self.ignore_iou_thresh:
                    continue
    
                # Find cell and calculate linear index
                scale_idx = torch.div(anchor, self.anchors_per_scale, rounding_mode='trunc')
                anchor_idx = anchor % len(self.SCALES)
                scaled_center_x, scaled_center_y = (bbox[0] + bbox[2] / 2) * self.SCALES[scale_idx], (bbox[1] + bbox[3] / 2) * self.SCALES[scale_idx]
                cell_x, cell_y = int(scaled_center_x), int(scaled_center_y)
                linear_idx = grid_to_linear(cell_x, cell_y, anchor_idx, scale_idx, self.anchors_per_scale, self.SCALES)
                # If anchor is taken
                if targets[linear_idx, 4] != 0:
                    continue

                w, h = torch.log(bbox[2]/self.anchors[anchor][0]), torch.log(bbox[3]/self.anchors[anchor][1])
                x, y = scaled_center_x - cell_x + w/2, scaled_center_y - cell_y + h/2

                if not anchor_found:
                    targets[linear_idx,:] = torch.tensor([x, y, w, h, 1, bbox[4]])
                    anchor_found = True
                else: # If anchor should be ignored
                    targets[linear_idx,4] = -1

        return img, targets

if __name__ == '__main__':
    import config
    from transforms import *
    import torchvision.transforms as T
    from utils import plot_sample

    t = T.Compose([
        Resize(),
        ToTensor(),
        Normalize()
    ])

    dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                        config.ANNOTATIONS_PATH, 
                        config.IMAGES_PATH,
                        config.ANCHORS,
                        transform=t)

    for image, targets in dataset:
        print(image)
        print(targets.shape)
        input()
