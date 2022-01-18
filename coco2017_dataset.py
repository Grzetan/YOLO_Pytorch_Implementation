import torch 
import torch.nn as nn
from torch.utils.data import Dataset
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


    def __len__(self):
        return len(self.annotations)

    # Function for dataloader
    def collate(self, batch):
        imgs = batch[0][0].unsqueeze(0)
        targets = batch[0][1]
        obj_mask = batch[0][2].unsqueeze(0)
        noobj_mask = batch[0][3].unsqueeze(0)
        for b in range(1, len(batch)):
            imgs = torch.cat((imgs, batch[b][0].unsqueeze(0)), dim=0)
            targets = torch.cat((targets, batch[b][1]), dim=0)
            obj_mask = torch.cat((obj_mask, batch[b][2].unsqueeze(0)), dim=0)
            noobj_mask = torch.cat((noobj_mask, batch[b][3].unsqueeze(0)), dim=0)

        return imgs, targets, obj_mask, noobj_mask

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

        # Read image
        img = Image.open(os.path.join(self.images_path, img_path)).convert('RGB')
        img = np.asarray(img)

        # Ensure that all bboxes have inside of the image
        bboxes_params[...,2:3][bboxes_params[...,2:3] > img.shape[1]] = img.shape[1]
        bboxes_params[...,3:4][bboxes_params[...,3:4] > img.shape[0]] = img.shape[0]

        # Convert to x, y, w, h
        bboxes_params[...,2:3] = bboxes_params[...,2:3] - bboxes_params[...,0:1]
        bboxes_params[...,3:4] = bboxes_params[...,3:4] - bboxes_params[...,1:2]
        # Ensure that all bboxes have width and height of at least 1
        bboxes_params[...,2:4][bboxes_params[...,2:4] == 0] = 1

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes = bboxes_params)
            img = transformed['image']
            bboxes_params = transformed['bboxes']

        if not isinstance(bboxes_params, torch.Tensor):
            bboxes_params = torch.tensor(bboxes_params)


        # Build targets
        n_bboxes = sum([self.anchors_per_scale * S * S for S in self.SCALES]) # Total number of bboxes
        bboxes_params[...,0:4] /= self.IMG_SIZE

        obj_mask = torch.zeros(n_bboxes, dtype=torch.bool)
        noobj_mask = torch.ones(n_bboxes, dtype=torch.bool)
        targets = torch.zeros((bboxes_params.shape[0], 6)) # x, y, w, h, objetness, class_label

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
                if obj_mask[linear_idx]:
                    continue

                x, y = scaled_center_x - cell_x, scaled_center_y - cell_y
                w, h = torch.log(bbox[2]/self.anchors[anchor][0]), torch.log(bbox[3]/self.anchors[anchor][1])

                if not anchor_found:
                    targets[k][:] = torch.tensor([x, y, w, h, 1, bbox[4]])
                    obj_mask[linear_idx] = True
                    noobj_mask[linear_idx] = False
                    anchor_found = True
                else: # If anchor should be ignored
                    noobj_mask[linear_idx] = False

        return img, targets, obj_mask, noobj_mask

if __name__ == '__main__':
    import config
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    f = open(os.path.join(config.DATASET_PATH, 
                          config.TRAIN_PATH,
                          config.CLASSES_PATH), 'r')
    class_names = [cls.strip() for cls in f.readlines()]

    dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                       config.ANNOTATIONS_PATH, 
                       config.IMAGES_PATH,
                       config.ANCHORS,
                       transform=config.TRANSFORMS)

    anchors = [(a[0]/config.IMG_SIZE, a[1]/config.IMG_SIZE) for a in config.ANCHORS]

    for i in range(len(dataset)):
        img, targets = dataset[i]
        print('\r', i, '/', len(dataset), end='')

    # for i in range(61, 110):
    #     img, anchors_params, ignore_indices = dataset[i]
        
    #     fig, ax = plt.subplots()
    #     ax.imshow(img)

    #     for anchor in anchors_params:
    #         cell_x, cell_y, anchor_idx, scale_idx = linear_to_grid(int(anchor[-1]), 3, config.SCALES)
    #         grid_size = config.IMG_SIZE / config.SCALES[scale_idx]
    #         w = anchors[scale_idx * 3 + anchor_idx][0] * torch.exp(anchor[2]) * config.IMG_SIZE
    #         h = anchors[scale_idx * 3 + anchor_idx][1] * torch.exp(anchor[3]) * config.IMG_SIZE
    #         x = (cell_x * grid_size + anchor[0] * grid_size) - w/2
    #         y = (cell_y * grid_size + anchor[1] * grid_size) - h/2
    #         rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
    #         ax.add_patch(rect)
    #     plt.show()

