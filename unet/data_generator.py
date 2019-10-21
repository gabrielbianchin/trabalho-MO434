import numpy as np
import keras
import cv2
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df_img, df_mask, load_memory=False, preprocessing = 'div', aug_flag=False, batch_size=4, dim=(384, 512), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.n_channels = n_channels
        self.df_img = df_img
        self.df_mask = df_mask
        self.preprocessing = preprocessing
        self.mean = [31.74609085, 42.01666982, 48.74235915]
        self.std = [9.77630045, 13.52184419, 16.22450647]
        self.load_memory = load_memory
        self.aug_flag = aug_flag
        self.list_IDs = df_img.index.values
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
        if self.aug_flag == True:
            self.augmentation = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.CropAndPad(
                    percent=(-0.08, 0.08),
                    pad_mode=["symmetric"]),
                iaa.Add((-20, 20)),
                iaa.AddToHueAndSaturation((-15, 15)),
                iaa.Affine(
                    rotate=(-3, 3),
                    mode=["symmetric"])
            ])
        else:
            self.augmentation = None

        if self.load_memory == True:
            self.imgs = np.empty((len(self.df_img), *self.dim, self.n_channels), dtype=np.float16)
            self.masks = np.empty((len(self.df_img), *self.dim, self.n_channels), dtype=np.uint8)
            self.load_listID()
            
            
    def load_listID(self):
        for i, ID in enumerate(self.list_IDs):
            self.imgs[i] = load_img(self.df_img.loc[ID]['path'], target_size=self.dim)
            self.masks[i] = load_img(self.df_mask.loc[ID]['path'], target_size=self.dim)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)
        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def __data_generation(self, list_IDs_temp, indexes):
        if self.load_memory == True:
            self.X = self.imgs[indexes]
            self.y = self.masks[indexes]
        else:
            self.X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float16)
            self.y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
            for i, ID in enumerate(list_IDs_temp):
                self.X[i] = load_img(self.df_img.loc[ID]['path'], target_size=self.dim)
                self.y[i] = load_img(self.df_mask.loc[ID]['path'], target_size=self.dim)
        if self.aug_flag == True:
            for i in range(len(self.X)):
                self.X[i], self.y[i] = self.augmentation(image=self.X[i], segmentation_maps=self.y[i])
                self.y[i,...,0] = np.bitwise_and(self.y[i,...,1] == 0, self.y[i,...,2] == 0)
        if self.preprocessing == 'norm':
            for i in range(len(self.X)):
                self.X[i] = cv2.GaussianBlur(self.X[i].astype('uint8'),(5,5),1,1)
                self.X[i] -= self.mean
                self.X[i] /= self.std
                self.X[i] = (1-(-1))/(self.X[i].max()-self.X[i].min())*(self.X[i]-self.X[i].max())+1
        if self.preprocessing == 'div':
            for i in range(len(self.X)):
                self.X[i] /= 255

        self.y[self.y==255] = 1
        self.y[np.bitwise_and(self.y!=0, self.y!=1)] = 0

        
        return self.X, self.y[...,1:]
