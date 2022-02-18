import torch
import torch.nn as nn
from utils import iou

class YOLOLoss(nn.Module):
    def __init__(self, anchors, scales, device):
        super(YOLOLoss, self).__init__()
        anchors = torch.tensor(anchors)
        self.anchors = torch.cat((
            anchors[0:3].repeat(scales[0]**2,1),
            anchors[3:6].repeat(scales[1]**2,1),
            anchors[6:9].repeat(scales[2]**2,1)
        ), dim=0).unsqueeze(0).to(device)
        self.scales = scales
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.noobj_lambda = 1
        self.obj_lambda = 2
        self.bbox_lambda = 2
        self.class_lambda = 1

    def forward(self, preds, targets):
        obj_mask = targets[...,4] == 1
        noobj_mask = targets[...,4] == 0
        anchors = self.anchors.repeat(preds.shape[0],1,1)

        bbox_preds = torch.cat((torch.sigmoid(preds[...,0:2][obj_mask]), torch.exp(preds[...,2:4][obj_mask]) * anchors[obj_mask]), dim=-1)
        ious = iou(bbox_preds, targets[...,0:4][obj_mask], format_='midpoint').squeeze()
        obj_loss = self.bce(preds[...,4][obj_mask], ious * targets[...,4][obj_mask])
        noobj_loss = self.bce(preds[...,4][noobj_mask], targets[...,4][noobj_mask])
        class_loss = self.ce(preds[...,5:][obj_mask], targets[...,5][obj_mask].long())
        bbox_loss = self.mse(preds[...,0:4][obj_mask], targets[...,0:4][obj_mask])

        loss = (self.obj_lambda * obj_loss + self.noobj_lambda * noobj_loss + 
               self.class_lambda * class_loss + self.bbox_lambda * bbox_loss)

        return loss