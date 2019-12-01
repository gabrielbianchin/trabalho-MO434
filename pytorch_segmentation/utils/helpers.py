import os
import torch
import torch.nn as nn
import numpy as np
import math
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt


def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
            center = factor - 1
    else:
            center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()

def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))
    
def plot_inference(model, dataloader, batches=1):
    invTrans = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                        std = [1/0.5, 1/0.5, 1/0.5]),
                                   transforms.Normalize(mean = [-0.5, -0.5, -0.5], 
                                                        std = [1., 1., 1.]),])
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            y_pred = model(X)

            for j in range(len(X)):
                f, ax = plt.subplots(1,3)
                f.set_figheight(15)
                f.set_figwidth(15)
                ax[0].imshow(invTrans(X[j]).permute(1, 2, 0).cpu().numpy())
                ax[1].imshow(y.permute(0, 1, 2).cpu().numpy()[j], vmin=0, vmax=2)
                ax[2].imshow(np.argmax(y_pred.permute(0, 2, 3, 1).cpu().numpy(), -1)[j], vmin=0, vmax=2)
                plt.show()

            if i == batches:
                break