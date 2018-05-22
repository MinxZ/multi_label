from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from my_classes import DataGenerator

# from load_data import *
# from m_model import *

# Load datasets
x_train, y_train, x_val, y_val = load_multi_label_data()
width = x_val.shape[1]
n_class = y_val.shape[1]


# Parameters
params = {'dim': (224, 224, 3),
          'batch_size': 32,
          'n_classes': 223,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition =  # IDs
labels =  # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)


# Loading model
model_name = 'Xception'
MODEL = Xception
batch_size = 64
model = build_model(MODEL, width, n_class)


# Load weights
print('\n Loading weights. \n')
try:
    model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)
except:
    print(' Train fc layer firstly.\n')
    fc_model(MODEL, x_train, batch_size)
print(f' Load fc_{model_name}.h5 successfully.\n')

model.load_weights('../models/Xception_69_256.h5', by_name=True)
model.load_weights(f'../models/fc_{model_name}.h5', by_name=True)

# Compile model
optimizer = 'SGD'
lr = 5e-4  # 1-5e4
epoch = 1e4
reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1
angle = 20

print("  Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
model.compile(
    loss='binary_crossentropy',
    optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
    metrics=['binary_accuracy'])

# datagen and val_datagen
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # preprocessing_function=get_random_eraser(
    #     p=0.2, v_l=0, v_h=255, pixel_level=True),  # 0.1-0.4
    rotation_range=20,  # 10-30
    width_shift_range=0.2,  # 0.1-0.3
    height_shift_range=0.2,  # 0.1-0.3
    shear_range=0.2,  # 0.1-0.3
    zoom_range=0.2,  # 0.1-0.3
    horizontal_flip=True,
    fill_mode='nearest')
# val_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=0.3, patience=reduce_lr_patience, verbose=2)

# Start fitting model
print(" Fine tune " + model_name + ": \n")
# Design model
model = Sequential()
[...]  # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, checkpointer, reduce_lr])
