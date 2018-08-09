from __future__ import absolute_import, division, print_function

import argparse
import multiprocessing as mp
import random

import numpy as np
import tensorflow as tf
from keras import backend
from keras.applications import *
from keras.backend.common import (epsilon, floatx, image_data_format,
                                  image_dim_ordering, set_image_dim_ordering)
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from image import *
from model import *

model_name = 'ResNet50'
optimizer = 2e-5
lr = 2e-5


def run(model_name, optimizer, lr):
    if model_name == "ResNet50":
        print('\n  For Resnet')
        from keras.applications.imagenet_utils import preprocess_input
    elif model_name[:-3] == "DenseNet":
        print('\n  For DenseNet')
        from keras.applications.densenet import preprocess_input
    else:
        print('\n  For model = tf')
        from keras.applications.inception_v3 import preprocess_input

    # Load datasets
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()
    epochs = 10000

    # Loading model
    model_config, fc, pred, layer_names, input_shape = load_model_config()
    MODEL = model_config[model_name][1]
    batch_size = model_config[model_name][0]

    def build_model():
        print('\n  Build model')
        h5_name = f'../models/{model_name}_{len(fc)}_fc.h5'
        checkpointer = ModelCheckpoint(
            filepath=h5_name, verbose=0, save_best_only=True)

        cnn_model = MODEL(
            include_top=False, input_shape=input_shape, weights='imagenet', pooling='avg')
        inputs = Input(shape=input_shape)
        x = cnn_model(inputs)
        model = tri_fc(inputs, x, fc, pred, layer_names)

        try:
            model.load_weights(
                f'../models/{len(fc)}_fc_{model_name}.h5', by_name=True)
            print('\n  Succeed on loading fc wight ')
        except:
            print('\n  Train fc')
            data_val = x_val
            split = int(data_val.shape[0] * 4 / 5)
            x_train_fc = data_val[:split]
            x_val_fc = data_val[split:]
            y_train_fc = []
            y_val_fc = []
            for x in range(3):
                y_train_fc.append(y_val[x][:split])
                y_val_fc.append(y_val[x][split:])

            fc_model_train(x_train_fc, y_train_fc, x_val_fc, y_val_fc, batch_size,
                           cnn_model, fc, pred, layer_names, model_name, preprocess_input)
            # fc_model_train(x_train, y_train, x_val, y_val, batch_size,
            #                cnn_model, fc, pred, layer_names, model_name, preprocess_input)
            model.load_weights(
                f'../models/{len(fc)}_fc_{model_name}.h5', by_name=True)
        return model, checkpointer

    print('\n  Loading model')
    try:
        with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score}):
            model = load_model(
                f'../models/{model_name}_{len(fc)}_fc.h5')
        checkpointer = ModelCheckpoint(
            filepath=f'../models/{model_name}_{len(fc)}_fc_fine_tune.h5', verbose=0, save_best_only=True)
        print('\n  Ready to fine tune.')
    except:
        model, checkpointer = build_model()

    # callbacks

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # preprocessing_function=get_random_eraser( p=0.2, v_l=0, v_h=255, pixel_level=True),
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    activation_2 = ['softmax', 'sigmoid', 'sigmoid']
    for i in range(2):
        if i == 0:
            lossWeights = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': 0.3,
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': 10,
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 10}
            losses = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': "categorical_crossentropy",
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': 'binary_crossentropy',
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 'binary_crossentropy'}
            metrics = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': ["categorical_accuracy"],
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': [f1_score],
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': [f1_score]}
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=6, verbose=2, mode='auto')
            reduce_lr = ReduceLROnPlateau(
                factor=np.sqrt(0.1), patience=3, verbose=2)
            opt = Adam(lr=lr)
            print(f"\n  {model_name}: Optimizer=" +
                  optimizer + " lr=" + str(lr) + " \n")
        elif i == 1:
            lossWeights = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': 0.2,
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': 10,
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 10}
            losses = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': "categorical_crossentropy",
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': f1_loss,
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': 'binary_crossentropy'}
            metrics = {
                f'{layer_names[0]}_{activation_2[0]}_{pred[0]}': ["categorical_accuracy"],
                f'{layer_names[1]}_{activation_2[1]}_{pred[1]}': [f1_score],
                f'{layer_names[2]}_{activation_2[2]}_{pred[2]}': [f1_score]}
            opt = SGD(lr=1e-5, momentum=0.9, nesterov=True)
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=6, verbose=2, mode='auto')
            reduce_lr = ReduceLROnPlateau(
                factor=np.sqrt(0.1), patience=3, verbose=2)
            checkpointer = ModelCheckpoint(
                filepath=f'../models/{model_name}_{len(fc)}_fc_fine_tune.h5', verbose=0, save_best_only=True)

        model.compile(optimizer=opt, loss=losses,
                      loss_weights=lossWeights, metrics=metrics)

        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size / 5,
            validation_data=val_datagen.flow(
                x_val, y_val, batch_size=batch_size),
            validation_steps=len(x_val) / batch_size / 2,
            epochs=epochs,
            callbacks=[early_stopping, checkpointer, reduce_lr],
            max_queue_size=20,
            workers=8,
            use_multiprocessing=True)

    quit()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="DenseNet169", type=str)
    parser.add_argument(
        "--optimizer", help="which optimizer to use", default="Adam", type=str)
    parser.add_argument(
        "--lr", help="learning rate", default=2e-5, type=float)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.optimizer, args.lr)
