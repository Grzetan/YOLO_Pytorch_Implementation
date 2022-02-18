import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn

def plot_sample(img, bboxes_params, class_names=None, format_='corners'):
    """
    bboxes_params shape - (n,5)
    Indexes at every instance mean
    1,2,3,4,5 = x1, y1, w, h, class_idx
    """

    fig, ax = plt.subplots()
    ax.imshow(img)
    print(len(bboxes_params))
    for i, box in enumerate(bboxes_params):
        if isinstance(box, torch.Tensor):
            box = box.cpu().detach().numpy()

        if format_ == 'midpoint':
            box[0] = box[0] - box[2] / 2
            box[1] = box[1] - box[3] / 2
        rect = patches.Rectangle(box[0:2], box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names is not None:
            ax.text(box[0], box[1], class_names[int(box[5])], fontsize=10)
    
    plt.show()

def iou(preds, labels, format_='corners', only_size=False):
    if only_size:
        assert preds.shape[-1] == 2
        assert labels.shape[-1] == 2
        x = torch.min(preds[...,0:1], labels[...,0:1])
        y = torch.min(preds[...,1:2], labels[...,1:2])
        intersection = x * y
        return intersection / (preds[...,0:1] * preds[...,1:2]
                               + labels[...,0:1] * labels[...,1:2] 
                               - intersection + 1e-8)
    
    if format_ == 'corners':
        preds_x1 = preds[...,0:1]
        preds_y1 = preds[...,1:2]
        preds_x2 = preds[...,2:3]
        preds_y2 = preds[...,3:4]
        labels_x1 = labels[...,0:1]
        labels_y1 = labels[...,1:2]
        labels_x2 = labels[...,2:3]
        labels_y2 = labels[...,3:4]
    else:
        preds_x1 = preds[...,0:1] - preds[...,2:3] / 2
        preds_y1 = preds[...,1:2] - preds[...,3:4] / 2
        preds_x2 = preds[...,0:1] + preds[...,2:3] / 2
        preds_y2 = preds[...,1:2] + preds[...,3:4] / 2
        labels_x1 = labels[...,0:1] - labels[...,2:3] / 2
        labels_y1 = labels[...,1:2] - labels[...,3:4] / 2
        labels_x2 = labels[...,0:1] + labels[...,2:3] / 2
        labels_y2 = labels[...,1:2] + labels[...,3:4] / 2

    x1 = torch.max(preds_x1, labels_x1)
    y1 = torch.max(preds_y1, labels_y1)
    x2 = torch.min(preds_x2, labels_x2)
    y2 = torch.max(preds_y2, labels_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    preds_area = (preds_x2 - preds_x1) * (preds_y2 - preds_y1)
    labels_area = (labels_x2 - labels_x1) * (labels_y2 - labels_y1)

    return intersection / (preds_area + labels_area - intersection + 1e-6)

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

def nms(output, objetness_thresh=0.4, iou_ignore_thresh=0.7):
    detections = [d for d in output if d[4] > objetness_thresh]
    detections = sorted(detections, key=lambda x: x[...,4], reverse=True)
    after_nms = []

    while detections:
        bbox = detections.pop(0).to(output.device)
        detections = [box for box in detections if box[5] == bbox[5] and iou(box[:4], bbox[:4], format_='midpoint') < iou_ignore_thresh]
        after_nms.append(bbox)
    return after_nms
