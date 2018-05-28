from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from image import ImageDataGenerator, resizeAndPad
from load_data import *
from model import *

# Load datasets
x_train, y_train, x_val, y_val = load_multi_label_data('../data/json')
width = 224
n_class = y_val.shape[1]
n = x_train.shape[0]

batch_size_model = {'MobileNet': (48, MobileNet), 'Xception': (
    16, Xception), 'InceptionResNetV2': (16, InceptionResNetV2)}

model_name = 'MobileNet'
# model_name = 'Xception'
# model_name = 'InceptionResNetV2'

batch_size, MODEL = batch_size_model[model_name]
model = build_model(MODEL, width, n_class, model_name, batch_size)
# Load weights

"""
model_name = 'Xception'
MODEL = Xception
batch_size = 16
with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score, 'precision': precision, 'recall': recall}):
    model = load_model('../models/Xception_f1.h5')
"""
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

losses = {'f1': f1_loss, 'bc': 'binary_crossentropy'}
configs = [('bc', Adam(lr=1e-4)), ('f1', Adam(lr=1e-4)),
           ('f1', SGD(lr=1e-4, momentum=0.9, nesterov=True))]
for i, config in enumerate(configs):
    print(f'{i + 1} trial')
    loss_name, opt = config
    reduce_lr_patience = 2
    patience = 5  # reduce_lr_patience+1 + 1
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=patience, verbose=2, mode='auto')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/{model_name}_{loss_name}.h5', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(
        factor=np.sqrt(0.1), patience=reduce_lr_patience, verbose=2)

    model.compile(
        loss=losses[loss_name],
        optimizer=opt,
        metrics=[f1_score, precision, recall])
    # Start fitting model
    fold = 20
    epoch = 1e4
    print(" Fine tune " + model_name + ": \n")
    model.fit_generator(
        datagen.flow(x_train, '../data/train_data', width,
                     y_train, batch_size=batch_size),
        steps_per_epoch=len(x_train) / batch_size / fold,
        validation_data=val_datagen.flow(
            x_val, '../data/val_data', width, y_val, batch_size=batch_size),
        validation_steps=len(x_val) / batch_size,
        epochs=epoch,
        callbacks=[early_stopping, reduce_lr, checkpointer],
        workers=4)
