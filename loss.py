import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.noobj_lambda = 1
        self.obj_lambda = 1
        self.bbox_lambda = 1
        self.class_lambda = 1

    def forward(self, preds, targets, obj_mask, noobj_mask):
        # print(preds.shape)
        # print(preds[obj_mask].shape)
        # print(targets.shape)
        return preds