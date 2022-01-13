import torch 
import torch.nn as nn
import os
import config

class Route(nn.Module):
    def __init__(self, start, end):
        self.start = start
        self.end = end

class YOLOHead(nn.Module):
    def __init__(self, scale_idx):
        self.scale_idx = scale_idx

class YOLO(nn.Module):
    def __init__(self, cfgfile):
        super(YOLO, self).__init__()
        f = open(cfgfile, 'r')
        lines = [line.strip() for line in f.readlines() if line[0] != '#' and not line.startswith('\n')]
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
        modules = nn.ModuleList()
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
                pad = int(block['pad'])
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
                
                modules.append(module)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                module.add_module(f'upsample_{i}', nn.Upsample(scale_factor=stride))
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                start = int(layers[0])
                if len(layers) > 1:
                    end = int(layers[1])
                else:
                    end = 0

                if not end:
                    filters = output_filters[i+start]
                else:
                    filters = output_filters[i+start] + output_filters[end]

                module.add_module(f'route_{i}', Route(start, end))
            elif block['type'] == 'yolo':
                module.add_module(f'yolo_{i}', YOLOHead(scale_idx))

            output_filters.append(filters)
            prev_filters = filters
            modules.append(module)

        return modules

if __name__ == '__main__':
    yolo = YOLO(config.YOLOV3_FILE)