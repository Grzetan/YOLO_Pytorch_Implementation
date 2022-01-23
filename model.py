import torch 
import torch.nn as nn
import os
import config
import numpy as np

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
    def __init__(self, grid_size, anchors_per_scale, n_classes, anchors=None):
        super(YOLOHead, self).__init__()
        self.n_classes = n_classes
        self.anchors_per_scale = anchors_per_scale
        self.anchors = anchors
        self.stride = config.IMG_SIZE / grid_size
        if anchors is not None:
            self.anchors = torch.tensor(anchors).repeat(grid_size*grid_size,1).unsqueeze(0)
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)
        x_offset = torch.tensor(a).view(-1,1)
        y_offset = torch.tensor(b).view(-1,1)
        self.offset = torch.cat((x_offset, y_offset), 1).repeat(1,self.anchors_per_scale).view(-1,2).unsqueeze(0)

    def forward(self, X):
        batch_size = X.shape[0]
        bbox_attrs = 5+self.n_classes
        grid_size = X.shape[-1]
        X = X.view(batch_size, bbox_attrs*self.anchors_per_scale, grid_size*grid_size)
        X = X.transpose(1,2).contiguous()
        X = X.view(batch_size, grid_size*grid_size*self.anchors_per_scale, bbox_attrs)
        if not self.training and self.anchors is not None:
            x = torch.zeros((X.shape[0], X.shape[1], 6)).to(X.device)
            x[...,0:2] = torch.sigmoid(X[...,0:2])
            x[...,4] = torch.sigmoid(X[...,4])
            X[...,5:] = torch.sigmoid(X[...,5:])
            offset = self.offset.to(X.device)
            x[...,0:2] += offset
            x[...,0:2] *= self.stride

            anchors = self.anchors.to(X.device)
            x[...,2:4] = torch.exp(X[...,2:4])*anchors
            x[...,5] = torch.argmax(X[...,5:],dim=2)
            return x
        else:
            return X

class YOLO(nn.Module):
    def __init__(self, cfgfile, n_classes, anchors_per_scale, scales, anchors):
        super(YOLO, self).__init__()
        self.n_classes = n_classes
        self.scales = scales
        self.anchors = anchors
        self.anchors_per_scale = anchors_per_scale
        f = open(cfgfile, 'r')
        lines = [line.strip() for line in f.readlines() if line[0] != '#' and not line.startswith('\n')]
        # This contains indexes of modules which outputs will
        # be used in shortcut or route layers
        self.needed_outputs = []
        blocks = self.create_blocks(lines)
        self.info = blocks[0]
        self.blocks = blocks[1:]
        self.create_modules()

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
        self.layers = nn.ModuleList()
        prev_filters = 3
        output_filters = []
        scale_idx = 0
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
                # Change number of channels in the layer before yolo head to match number of classes
                self.layers[-1][0].out_channels = (5+self.n_classes) * self.anchors_per_scale
                module.add_module(f'yolo_{i}', YOLOHead(self.scales[scale_idx] ,self.anchors_per_scale, self.n_classes, 
                                                        self.anchors[scale_idx*self.anchors_per_scale:(scale_idx+1)*self.anchors_per_scale]))
                scale_idx += 1

            output_filters.append(filters)
            prev_filters = filters
            self.layers.append(module)

    def forward(self, X):
        outputs = {}
        predictions = None

        for i, module in enumerate(self.layers):
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

    def load_weights(self, file):
        f = open(file, 'rb')

        header = np.fromfile(f, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = header[3]

        weights = np.fromfile(f, dtype=np.float32)
        ptr=0
        for i in range(len(self.layers)):
            type_ = self.blocks[i]['type']

            if type_ == 'convolutional':
                model = self.layers[i]
                try:
                    batch_normalize = int(self.blocks[i]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def to(self, device):
        for module in self.layers:
            module.to(device)