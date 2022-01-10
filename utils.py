import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_sample(img, bboxes_params, class_names=None):
    # bboxes_params shape - (n,5)
    # Indexes at every instance mean
    # 1,2,3,4,5 = x1, y1, x2, y2, class_idx
    
    fig, ax = plt.subplots()
    ax.imshow(img)

    for i, box in enumerate(bboxes_params):
        rect = patches.Rectangle(box[0:2], box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names is not None:
            ax.text(box[0], box[1], class_names[int(box[4])], fontsize=10)
    
    plt.show()