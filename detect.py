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

raw_dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                   config.ANNOTATIONS_PATH, 
                   config.IMAGES_PATH,
                   config.ANCHORS,
                   transform=config.RAW_TRANSFORMS)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'gpu')
model = YOLO(config.YOLOV3_FILE, config.N_CLASSES, config.ANCHORS_PER_SCALE, config.SCALES, config.ANCHORS)
# model.load_state_dict(torch.load(config.MODEL_PATH))
model.load_weights(config.WEIGHTS_PATH)
model.to(device)
model.eval()

for i in range(0,len(dataset)):
    img, targets = dataset[i]
    img2, targets2 = raw_dataset[i]
    start = time.time()

    img = img.to(device).unsqueeze(0)

    output = model(img)
    detections = nms(output[0])
    print(time.time() - start)
    plot_sample(img2, detections, format_='midpoint', class_names=config.CLASS_NAMES)
