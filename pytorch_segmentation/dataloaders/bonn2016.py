from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class BonnDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 3
        self.palette = palette.CityScpates_palette
        print(kwargs)
        super(BonnDataset, self).__init__(**kwargs)

    def _set_files(self):
        assert (self.split in ['train', 'val'])
        image_paths = pd.read_pickle(self.root + '/X_' + self.split + '.pkl')['path']
        label_paths = pd.read_pickle(self.root + '/y_' + self.split + '.pkl')['path']
        self.files = list(zip(image_paths, label_paths))
        self.ids = image_paths.index

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = self.ids[index]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        mask = np.asarray(Image.open(label_path), dtype=np.int32)

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented['image']
        #     original_mask = augmented['mask']
            
        label = np.zeros_like(mask[..., 0], dtype=np.int32)
        label[mask[..., 1] == 255] = 1
        label[mask[..., 0] == 255] = 2
        
        return image, label, image_id


class Bonn2016(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, shuffle=False,
                    flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False, val=False):

        self.MEAN = [0.5, 0.5, 0.5]
        self.STD = [0.5, 0.5, 0.5]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = BonnDataset(**kwargs)
        super(Bonn2016, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)