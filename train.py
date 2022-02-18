import torch 
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from coco2017_dataset import COCO2017
from loss import YOLOLoss
from model import YOLO
import config
import os
import time

def train(model, criterion, optimizer, train_loader, val_loader, scaler, device, epochs=1, save=100):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        losses = []
        start = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)
            # Use mixed precision training
            with autocast():
                output = model(imgs)
                loss = criterion(output, targets)
            # Step all components
            scaler.scale(loss).backward()
            # Unscale gradients for gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            # Update weights
            scaler.step(optimizer)
            # Update scaler
            scaler.update()
            # Bookkeeping
            losses.append(loss.item())
            print(f'\rEpoch {epoch+1}/{epochs}, Progress: {round(i/len(train_loader) * 100, 3)}%, Mean Loss: {round(sum(losses) / len(losses), 5)}, \
                    Loss: {round(loss.item(), 5)}, Saved {i//save} times, Time Elapsed: {(time.time() - start) // 60}min', end='')       
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(" ")
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO(config.YOLOV3_FILE, config.N_CLASSES, config.ANCHORS_PER_SCALE, config.SCALES, config.ANCHORS)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
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
train_loader = train_dataset.get_loader(batch_size=6)
val_loader = val_dataset.get_loader(batch_size=6)

scaler = GradScaler()

train(model, criterion, optimizer, val_loader, val_loader, scaler, device, epochs=100)