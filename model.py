import torch 
import torch.nn as nn
import os
import config

class Route(nn.Module):
    def __init__(self, start, end):
        super(Route, self).__init__()
        self.start = start
        self.end = end

class Shortcut(nn.Module):
    def __init__(self, from_):
        super(Shortcut, self).__init__()
        self.from_ = from_

class YOLOHead(nn.Module):
    def __init__(self, anchors_per_scale, n_classes):
        super(YOLOHead, self).__init__()
        self.n_classes = n_classes
        self.anchors_per_scale = anchors_per_scale
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        batch_size = X.shape[0]
        bbox_attrs = 5+self.n_classes
        grid_size = X.shape[-1]
        X = X.view(batch_size, bbox_attrs*self.anchors_per_scale, grid_size*grid_size)
        X = X.transpose(1,2).contiguous()
        X = X.view(batch_size, grid_size*grid_size*self.anchors_per_scale, bbox_attrs)
        X[...,:2] = torch.sigmoid(X[...,:2])
        X[...,4] = torch.sigmoid(X[...,4])
        X[...,5:] = self.softmax(X[...,5:])
        return X

class YOLO(nn.Module):
    def __init__(self, cfgfile, n_classes, anchors_per_scale):
        super(YOLO, self).__init__()
        self.device = device
        self.n_classes = n_classes
        self.anchors_per_scale = anchors_per_scale
        f = open(cfgfile, 'r')
        lines = [line.strip() for line in f.readlines() if line[0] != '#' and not line.startswith('\n')]
        # This contains indexes of modules which outputs will
        # be used in shortcut or route layers
        self.needed_outputs = []
        blocks = self.create_blocks(lines)
        self.info = blocks[0]
        self.blocks = blocks[1:]
        self.modules = self.create_modules()

    def create_blocks(self, lines):
        blocks = []
        block = {}

        for line in lines:
            if line[0] == '[':
                blocks.append(block)
                block = {}
                block['type'] = line[1:-1]
                continue

            key, val = line.split('=')
            key = key.strip()
            val = val.strip()
            block[key] = val

        blocks.append(block)

        return blocks[1:]

    def create_modules(self):
        modules = []
        prev_filters = 3
        output_filters = []
        scale_idx = 0

        for i, block in enumerate(self.blocks):
            module = nn.Sequential()

            if block['type'] == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                activation = block['activation']
                pad = (kernel_size - int(block['pad'])) // 2
                bias = True
                bn = False

                if 'batch_normalize' in block:
                    bias = False
                    bn = True

                module.add_module(f'conv_{i}', nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias))

                if bn:
                    module.add_module(f'bn_{i}', nn.BatchNorm2d(filters))
                
                if activation == 'leaky':
                    module.add_module(f'leaky_{i}', nn.LeakyReLU(0.1))
                
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                module.add_module(f'upsample_{i}', nn.Upsample(scale_factor=stride))
            elif block['type'] == 'shortcut':
                from_ = i + int(block['from'])
                self.needed_outputs.append(from_)
                module.add_module(f'shortcut_{i}', Shortcut(from_))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                start =  i + int(layers[0])
                if len(layers) > 1:
                    end = int(layers[1])
                else:
                    end = 0

                if not end:
                    filters = output_filters[start]
                    self.needed_outputs.append(start)
                else:
                    filters = output_filters[start] + output_filters[end]
                    self.needed_outputs.append(end)
                module.add_module(f'route_{i}', Route(start, end))
            elif block['type'] == 'yolo':
                module.add_module(f'yolo_{i}', YOLOHead(self.anchors_per_scale, self.n_classes))

            output_filters.append(filters)
            prev_filters = filters
            module.to(self.device)
            modules.append(module)

        return modules

    def forward(self, X):
        outputs = {}
        predictions = None

        for i, module in enumerate(self.modules):
            type_ = self.blocks[i]['type']
            if type_ == 'convolutional' or type_ == 'upsample':
                X = module(X)
            elif type_ == 'shortcut':
                X = X + outputs[module[0].from_]
            elif type_ == 'route':
                start = module[0].start
                end = module[0].end

                if not end:
                    X = outputs[start]
                else:
                    X = torch.cat((X, outputs[end]), 1)
            elif type_ == 'yolo':
                pred = module(X)
                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred), 1)
            if i in self.needed_outputs:
                outputs[i] = X
        return predictions

    def to(self, device):
        for module in self.modules:
            module.to(device)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo = YOLO(config.YOLOV3_FILE, config.N_CLASSES, config.ANCHORS_PER_SCALE)
    yolo.to(device)
    from coco2017_dataset import COCO2017
    from torch.utils.data import DataLoader
    dataset = COCO2017(os.path.join(config.DATASET_PATH, config.TRAIN_PATH), 
                       config.ANNOTATIONS_PATH, 
                       config.IMAGES_PATH,
                       config.ANCHORS,
                       transform=config.TRANSFORMS)

    loader = DataLoader(dataset, collate_fn=dataset.collate, batch_size=1)
    import time
    start = time.time()

    for i in loader:
        imgs, targets, obj_mask, noobj_mask = i
        imgs = imgs.to(device)
        targets = targets.to(device)
        obj_mask = obj_mask.to(device)
        noobj_mask = noobj_mask.to(device)
        break
    from loss import YOLOLoss
    loss = YOLOLoss()
    output = yolo(imgs)
    loss(output, targets, obj_mask, noobj_mask)
    print(time.time() - start)
    print(output.shape)
