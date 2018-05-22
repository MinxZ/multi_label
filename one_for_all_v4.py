from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import random

import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.preprocessing import image
from keras.preprocessing.image import *
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm


def tri_fc(x, fc_1, fc_2, fc_3, pred_1=25, pred_2=18, pred_3=7):
    processed_a = Dropout(0.5)(x)
    processed_a = Dense(fc_1, activation='elu',
                        name='processed_a_fc_1')(processed_a)
    processed_a = Dropout(0.5)(processed_a)
    processed_a = Dense(pred_1, activation='softmax',
                        name='processed_a_predictions')(processed_a)

    processed_b = Dropout(0.5)(x)
    processed_b = Dense(fc_2, activation='elu',
                        name='processed_b_fc_1')(processed_b)
    processed_b = Dropout(0.5)(processed_b)
    processed_b = Dense(pred_2, activation='softmax',
                        name='processed_b_predictions')(processed_b)

    processed_c = Dropout(0.5)(x)
    processed_c = Dense(fc_3, activation='elu',
                        name='processed_c_fc_1')(processed_c)
    processed_c = Dropout(0.5)(processed_c)
    processed_c = Dense(pred_3, activation='softmax',
                        name='processed_c_predictions')(processed_c)

    model = Model(inputs, [processed_a, processed_b, processed_c])

    return model


def tri_fc_featurens(x, fc_1, fc_2, fc_3):
    processed_a = Dropout(0.5)(x)
    processed_a = Dense(fc_1, activation='elu',
                        name='processed_a_fc_1')(processed_a)
    processed_b = Dropout(0.5)(x)
    processed_b = Dense(fc_2, activation='elu',
                        name='processed_b_fc_1')(processed_b)
    processed_c = Dropout(0.5)(x)
    processed_c = Dense(fc_3, activation='elu',
                        name='processed_c_fc_1')(processed_c)
    model = Model(inputs, [processed_a, processed_b, processed_c])

    x = concatenate([processed_a, processed_b, processed_c])
    model = Model(inputs, x)

    return model


def tri_features(fc_1, fc_2, fc_3):
    from keras.applications.inception_v3 import preprocess_input

    model_name = 'Xception'
    MODEL = Xception
    batch_size = 64

    width = 224
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
    x = cnn_model(x)
    model = tri_fc_featurens(x, fc_1, fc_2, fc_3)

    model.load_weights('../models/Xception_69_256.h5', by_name=True)
    model.load_weights(f'../models/tri.h5', by_name=True)
    # model.load_weights('../models/Xception_tri.h5', by_name=True)

    batch_size = 64
    for train_test in ['train', 'test']:
        X = np.load(f'../data/X_{train_test}.npy')
        features = model.predict(X, batch_size=batch_size, verbose=1)
        np.save(f'../data/Xception_tri_384_{train_test}', features)


def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    theta = np.deg2rad(np.random.uniform(-rg, rg))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def preprocess_image_worker_train(i):
    return random_rotation(x_train[i] / 127.5 - 1, 10)


def preprocess_image_worker_train2(i):
    return x_train[i] / 127.5 - 1


def preprocess_image_worker_val(i):
    return x_val[i] / 127.5 - 1


def preprocess_label_worker_train(i):
    return [y_train_1[i:i + batch_size], y_train_2[i:i + batch_size], y_train_3[i:i + batch_size]]


def preprocess_label_worker_val(i):
    return [y_val_1[i:i + batch_size], y_val_2[i:i + batch_size], y_val_3[i:i + batch_size]]



# Load datasets
dir = '../data/'
y_val = np.load(f'{dir}/y_val.npy')
x_val = np.load(f'{dir}/x_val.npy')
y_train = np.load(f'{dir}/y_train.npy')
x_train = np.load(f'{dir}/x_train.npy')

width = x_val.shape[1]
n_class = y_val.shape[1]
y_train_1, y_train_2, y_train_3 = y_train[:,
                                          :25], y_train[:, 25:25 + 18], y_train[:, 25 + 18:]
y_val_1, y_val_2, y_val_3 = y_val[:,
                                  :25], y_val[:, 25:25 + 18], y_val[:, 25 + 18:]


# Loading model
model_name = 'Xception'
MODEL = Xception
batch_size = 64

inputs = Input((width, width, 3))
cnn_model = MODEL(
    include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
x = cnn_model(inputs)

fc_1, fc_2, fc_3 = 512 // 8 * np.array((3, 4, 1))
model = tri_fc(x, fc_1, fc_2, fc_3)

model.load_weights('../models/Xception_69_256.h5', by_name=True)
model.load_weights(f'../models/tri.h5', by_name=True)

# Compile model
optimizer = 'SGD'
lr = 5e-4  # 1-5e4
epoch = 1e4
reduce_lr_patience = 5
patience = 10  # reduce_lr_patience+1 + 1
angle = 20

print("  Optimizer=" + optimizer + " lr=" + str(lr) + " \n")
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=lr, momentum=0.9, nesterov=True),
    metrics=['categorical_accuracy'])

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', patience=patience, verbose=2, mode='auto')
checkpointer = ModelCheckpoint(
    filepath=f'../models/{model_name}_tri1.h5', verbose=0, save_best_only=True)
reduce_lr = ReduceLROnPlateau(
    factor=0.3, patience=reduce_lr_patience, verbose=2)


def next_i(i, num, batch_size):
    i += batch_size
    if i > num - batch_size - 1:
        i = 0
    print(i)
    return i


def batch_generator(pool, x_data, preprocess_image_worker, preprocess_label_worker):
    i = -batch_size
    num = x_data.shape[0]
    while True:
        i = next_i(i, num, batch_size)
        imgs = pool.map(preprocess_image_worker, [
                        x for x in range(i, i + batch_size)])
        labs = preprocess_label_worker(i)
        yield np.array(imgs), labs


def get_generator(pool, x_data, preprocess_image_worker, preprocess_label_worker):
    gen = batch_generator(
        pool, x_data, preprocess_image_worker, preprocess_label_worker)
    return gen


pool = mp.Pool(processes=8)
train_gen = get_generator(
    pool, x_train, preprocess_image_worker_train, preprocess_label_worker_train)
val_gen = get_generator(
    pool, x_val, preprocess_image_worker_val, preprocess_label_worker_val)

model.fit_generator(
    train_gen,
    steps_per_epoch=len(x_train) / batch_size,
    validation_data=val_gen,
    validation_steps=len(x_val) / batch_size,
    epochs=epoch,
    callbacks=[early_stopping, checkpointer, reduce_lr])
