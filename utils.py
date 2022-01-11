import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def plot_sample(img, bboxes_params, class_names=None):
    """
    bboxes_params shape - (n,5)
    Indexes at every instance mean
    1,2,3,4,5 = x1, y1, w, h, class_idx
    """
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i, box in enumerate(bboxes_params):
        rect = patches.Rectangle(box[0:2], box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names is not None:
            ax.text(box[0], box[1], class_names[int(box[4])], fontsize=10)
    
    plt.show()

def iou(pred, target, only_size=False):
    """
    Calculate IOU of two boxes
    Boxes are expected to have format (x1, y1, w, h).

    If only_size is true, function will calculate IOU 
    only by looking at width and height, so preds and targets 
    should have only width and height in last dimension.
    """
    if only_size:
        assert pred.shape[-1] == 2
        assert target.shape[-1] == 2
        x = torch.min(pred[...,0:1], target[...,0:1])
        y = torch.min(pred[...,1:2], target[...,1:2])
        intersection = x * y
        return intersection / (pred[...,0:1] * pred[...,1:2]
                               + target[...,0:1] * target[...,1:2] 
                               - intersection + 1e-8)
    else:
        assert pred.shape[-1] == 4
        assert target.shape[-1] == 4
        pred_x1 = pred[...,0:1]
        pred_x2 = pred[...,0:1] + pred[...,2:3]
        pred_y1 = pred[...,1:2]
        pred_y2 = pred[...,1:2] + pred[...,3:4]
        target_x1 = target[...,0:1]
        target_x2 = target[...,0:1] + target[...,2:3]
        target_y1 = target[...,1:2]
        target_y2 = target[...,1:2] + target[...,3:4]
        
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)

        intersection = (x2 - x1) * (y2 - y1)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)  
        return intersection / (pred_area + target_area - intersection + 1e-8)

def grid_to_linear(cell_x, cell_y, anchor_idx, scale_idx, anchors_per_scale, scales):
    """
    Calculates linear index for specific cell and anchor.
    """

    idx = 0
    for i in range(0,scale_idx):
        idx += scales[i] * scales[i] * anchors_per_scale

    y_stride = scales[scale_idx] * anchors_per_scale
    x_stride = anchors_per_scale
    idx += y_stride * cell_y + x_stride * cell_x + anchor_idx
    return idx

def linear_to_grid(idx, anchors_per_scale, scales):
    scale_idx = 0
    tmp = 0
    for S in scales:
        tmp += S*S*anchors_per_scale
        if idx > tmp:
            scale_idx += 1
        else:
            break

    for i in range(0, scale_idx):
        idx -= scales[i] * scales[i] * anchors_per_scale
    
    y_stride = scales[scale_idx] * anchors_per_scale
    x_stride = anchors_per_scale

    cell_y = idx // y_stride
    idx -= cell_y * y_stride
    cell_x = idx // x_stride 
    idx -= cell_x * x_stride 

    return cell_x, cell_y, idx, scale_idx 