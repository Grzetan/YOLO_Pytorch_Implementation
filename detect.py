import torch
from coco2017_dataset import COCO2017
from model import YOLO
import config
import os
from utils import nms, plot_sample
import time

dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                   config.ANNOTATIONS_PATH, 
                   config.IMAGES_PATH,
                   config.ANCHORS,
                   transform=config.TRANSFORMS)

device = torch.device('cuda:0' if not torch.cuda.is_available() else 'cpu')
model = YOLO(config.YOLOV3_FILE, config.N_CLASSES, config.ANCHORS_PER_SCALE, config.SCALES, config.ANCHORS)
loader = dataset.get_loader(batch_size=1)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
# model.load_weights(config.WEIGHTS_PATH)
model.to(device)
model.eval()

for i in range(0,len(dataset)):
    img, targets = dataset[i]
    org_img = img.squeeze().permute(1,2,0).numpy()
    start = time.time()
    img = img.to(device).unsqueeze(0)
    output = model(img)
    detections = nms(output[0])
    print(time.time() - start)
    plot_sample(org_img, detections, format_='midpoint', class_names=config.CLASS_NAMES2)
