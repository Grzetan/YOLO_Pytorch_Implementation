import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

DATASET_PATH = './COCO2017'
TRAIN_PATH = 'train'
VALID_PATH = 'valid'
IMAGES_PATH = ''
ANNOTATIONS_PATH = '_annotations.txt'
CLASSES_PATH = '_classes.txt'
YOLOV3_FILE = './cfg/yolov3.cfg'

IMG_SIZE = 608
SCALES = [13, 26, 52]

ANCHORS = [(10,13),  (16,30),  (33,23),  
           (30,61),  (62,45),  (59,119),  
           (116,90),  (156,198),  (373,326)]

N_CLASSES = 80
ANCHORS_PER_SCALE = 3

TRANSFORMS = A.Compose([
    A.Resize(width=IMG_SIZE, height=IMG_SIZE),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
], bbox_params=A.BboxParams(format='coco'))