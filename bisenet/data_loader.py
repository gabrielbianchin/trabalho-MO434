import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from visdom import Visdom

class SegDataset(Dataset):
    """Image Segmentation dataset loader."""

    def __init__(self, images_df, masks_df, shape, transform=None):
        """
        Args:
            images_path (string): Path to images.
            masks_path (string): Path to annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_df = images_df
        self.masks_df = masks_df
        self.transform = transform
        self.to_tensor = ToTensor()
        self.shape = shape

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.resize(cv2.imread(self.images_df.iloc[idx]['path']), self.shape), cv2.COLOR_BGR2RGB)
        original_mask = cv2.resize(cv2.imread(self.masks_df.iloc[idx]['path']), self.shape, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image = image, mask = original_mask)
            image = augmented['image']
            original_mask = augmented['mask']
            
        mask = np.zeros_like(original_mask[..., 0])
        mask[original_mask[..., 1] == 255] = 1
        mask[original_mask[..., 2] == 255] = 2

        return self.to_tensor(image).float(), torch.tensor(mask).long()