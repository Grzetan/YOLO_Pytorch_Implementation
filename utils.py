import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

def plot_sample(img, bboxes_params, class_names=None):
    """
    bboxes_params shape - (n,5)
    Indexes at every instance mean
    1,2,3,4,5 = x1, y1, x2, y2, class_idx
    """
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i, box in enumerate(bboxes_params):
        rect = patches.Rectangle(box[0:2], box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names is not None:
            ax.text(box[0], box[1], class_names[int(box[4])], fontsize=10)
    
    plt.show()

def iou(pred, target, only_size=False):
    """
    Calculate IOU of two boxes
    Boxes are expected to have format (x1, y1, x2, y2).

    If only_size is true, function will calculate IOU 
    only by looking at width and height, so preds and targets 
    should have only width and height in last dimension.
    """
    if only_size:
        assert pred.shape[-1] == 2
        assert pred.shape[-1] == 2
        x = torch.min(pred[...,0:1], target[...,0:1])
        y = torch.min(pred[...,1:2], target[...,1:2])
        intersection = x * y
        return intersection / (pred[...,0:1] * pred[...,1:2]
                               + target[...,0:1] * target[...,1:2] 
                               - intersection + 1e-8)
    else:
        assert pred.shape()[-1] == 4
        assert pred.shape()[-1] == 4
        x1 = torch.max(pred[...,0:1], target[...,0:1])
        y1 = torch.max(preds[...,1:2], target[...,1:2])
        x2 = torch.min(pred[...,2:3], target[...,2:3])
        y2 = torch.min(preds[...,3:4], target[...,3:4])

        intersection = (x2 - x1) * (y2 - y1)
        pred_area = (pred[...,2:3] - pred[...,0:1]) * (pred[...,3:4] * pred[1:2])
        target_area = (target[...,2:3] - target[...,0:1]) * (target[...,3:4] * target[1:2])  
        return intersection / (pred_area + target_area - intersection + 1e-8)