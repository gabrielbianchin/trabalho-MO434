import numpy as np
import glob
import multiprocessing
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras.backend as K
from keras.optimizers import *
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import History
from keras.models import load_model
import natsort
import pickle
from evaluation import define_metrics

from model import unet
from data_generator import DataGenerator


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)

    metrics_list = define_metrics()
    metrics_dict = {}
    for metric in metrics_list:
        metrics_dict[metric.__name__] = metric

    model = unet()
    model.compile(optimizer = Adam(lr = 1e-4),
              loss = 'binary_crossentropy',
              metrics = metrics_list)


    X_train = pd.read_pickle("../data/X_train.pkl")
    X_val = pd.read_pickle("../data/X_val.pkl")
    y_train = pd.read_pickle("../data/y_train.pkl")
    y_val = pd.read_pickle("../data/y_val.pkl")

    training_generator = DataGenerator(X_train, y_train, dataset_path='/mnt/1058CF1419A58A26/Bonn2016/', load_memory=False, preprocessing = 'div', aug_flag=False, batch_size=4)
    validation_generator = DataGenerator(X_val, y_val, dataset_path='/mnt/1058CF1419A58A26/Bonn2016/', load_memory=False, preprocessing = 'div',  aug_flag=False, batch_size=4)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    hist = History()

    model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=100,
                    workers=multiprocessing.cpu_count() - 1,
                    max_q_size=100,
                    callbacks=[es, mc, hist])

    with open('train_hist.pkl', 'wb') as f:
        pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
