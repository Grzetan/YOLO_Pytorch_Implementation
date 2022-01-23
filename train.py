import torch 
import torch.nn as nn
from coco2017_dataset import COCO2017
from loss import YOLOLoss
from model import YOLO
import config
import os

def train(model, criterion, optimizer, train_loader, val_loader, device, epochs=1):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        losses = []
        print(f'epoch {epoch}/{epochs}')
        for i, (imgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = model(imgs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f'\rLoss - {round(sum(losses) / len(losses),3)} {round((i / len(train_loader)) * 100, 3)}%', end='')
        torch.save(model.state_dict(), config.MODEL_PATH)
    

torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO(config.YOLOV3_FILE, config.N_CLASSES, config.ANCHORS_PER_SCALE)
criterion = YOLOLoss(config.ANCHORS, config.SCALES, device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
train_dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                    config.ANNOTATIONS_PATH, 
                    config.IMAGES_PATH,
                    config.ANCHORS,
                    transform=config.TRANSFORMS)
val_dataset = COCO2017(os.path.join(config.DATASET_PATH, config.VALID_PATH), 
                    config.ANNOTATIONS_PATH, 
                    config.IMAGES_PATH,
                    config.ANCHORS,
                    transform=config.TRANSFORMS)
train_loader = train_dataset.get_loader(batch_size=3)
val_loader = val_dataset.get_loader(batch_size=3)
train(model, criterion, optimizer, train_loader, val_loader, device)