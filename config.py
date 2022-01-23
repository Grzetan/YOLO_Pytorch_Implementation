import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

DATASET_PATH = './COCO2017'
TRAIN_PATH = 'train'
VALID_PATH = 'valid'
IMAGES_PATH = ''
ANNOTATIONS_PATH = '_annotations.txt'
CLASSES_PATH = '_classes.txt'
YOLOV3_FILE = './cfg/yolov3.cfg'
MODEL_PATH = './yolov3.pth'
WEIGHTS_PATH = './yolov3.weights'

IMG_SIZE = 416
SCALES = [13, 26, 52]

ANCHORS = [(116,90),  (156,198),  (373,326),  
           (30,61),  (62,45),  (59,119),  
           (10,13),  (16,30),  (33,23)]

N_CLASSES = 80
ANCHORS_PER_SCALE = 3

IOU_IGNORE_THRESH = 0.5
PROB_THRESH = 0.5

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

TRANSFORMS = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco'))

RAW_TRANSFORMS = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE)
], bbox_params=A.BboxParams(format='coco'))

CLASS_NAMES = [
    'aeroplane',
    'apple',
    'backpack',
    'banana',
    'baseball bat',
    'baseball glove',
    'bear',
    'bed',
    'ench',
    'bicycle',
    'bird',
    'boat',
    'book',
    'bottle',
    'bowl',
    'broccoli',
    'bus',
    'cake',
    'car',
    'carrot',
    'cat',
    'cell phone',
    'chair',
    'clock',
    'cow',
    'cup',
    'diningtable',
    'dog',
    'donut',
    'elephant',
    'fire hydrant',
    'fork',
    'frisbee',
    'giraffe',
    'hair drier',
    'handbag',
    'horse',
    'hot dog',
    'keyboard',
    'kite',
    'knife',
    'laptop',
    'microwave',
    'motorbike',
    'mouse',
    'orange',
    'oven',
    'parking meter',
    'person',
    'pizza',
    'pottedplant',
    'refrigerator',
    'remote',
    'sandwich',
    'scissors',
    'sheep',
    'sink',
    'skateboard',
    'skis',
    'snowboard',
    'sofa',
    'spoon',
    'sports ball',
    'stop sign',
    'suitcase',
    'surfboard',
    'teddy bear',
    'tennis racket',
    'tie',
    'toaster',
    'toilet',
    'toothbrush',
    'traffic light',
    'train',
    'truck',
    'tvmonitor',
    'umbrella',
    'vase',
    'wine glass',
    'zebra'
]