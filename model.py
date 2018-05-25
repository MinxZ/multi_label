import keras.backend as K
import numpy as np
from keras.applications import Xception
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *


def f1_loss(y_true, y_pred):

    TP = K.sum(y_pred * y_true)
    precision = TP / K.sum(y_true)
    recall = TP / K.sum(y_pred)
    f1 = (1 - 2 * precision * recall / (precision + recall))

    return f1


def f1_score(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def precision(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return precision


def recall(y_true, y_pred):
    y_pred = tf.round(y_pred)

    TP = tf.count_nonzero(y_pred * y_true)
    TN = tf.count_nonzero((y_pred - 1) * (y_true - 1))
    FP = tf.count_nonzero(y_pred * (y_true - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_true)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return recall

# index_array = np.random.permutation(n)[:6000]
# batch_x = np.zeros((len(index_array), width, width, 3))
# batch_y = y_train[index_array]
# for i, j in enumerate(tqdm(index_array)):
#     s_img = cv2.imread(f'../data/train_data/{j+1}.jpg')
#     b, g, r = cv2.split(s_img)       # get b,g,r
#     rgb_img = cv2.merge([r, g, b])     # switch it to rgb
#     x = resizeAndPad(rgb_img, (width, width))
#     batch_x[i] = x


# datagen and val_datagen
# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input,
#     # preprocessing_function=get_random_eraser(
#     #     p=0.2, v_l=0, v_h=255, pixel_level=True),  # 0.1-0.4
#     rotation_range=10,  # 10-30
#     width_shift_range=0.1,  # 0.1-0.3
#     height_shift_range=0.1,  # 0.1-0.3
#     shear_range=0.1,  # 0.1-0.3
#     zoom_range=0.1,  # 0.1-0.3
#     horizontal_flip=True,
#     fill_mode='nearest')
def get_features(MODEL, data, width, batch_size):
    cnn_model = MODEL(input_shape=(width, width, 3),
                      include_top=False,  weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    cnn_model = Model(inputs, x)
    # cnn_model.load_weights('../models/Xception_69_256.h5', by_name=True)

    features = cnn_model.predict(data, batch_size=batch_size, verbose=1)
    np.save(f'../data/fc_features_{model_name}', features)
    return features


def fc_model(MODEL, x_train, y_train, width, batch_size):
    try:
        features = np.load(f'../data/fc_features_{model_name}.npy')
    except:
        features = get_features(MODEL, x_train, width, 16)

    # Training fc models
    inputs = Input(features.shape[1:])
    x = inputs
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model_fc = Model(inputs, x)

    model_fc.compile(
        optimizer='adam',
        # loss='binary_crossentropy',
        loss=f1_loss,
        metrics=[f1_score, precision, recall])
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(
        filepath=f'../models/fc_{model_name}.h5', verbose=0, save_best_only=True)
    model_fc.fit(
        features,
        y_train,
        batch_size=128,
        epochs=10000,
        validation_split=0.1,
        callbacks=[checkpointer, early_stopping])


def build_model(MODEL, width, n_class):
    print(' Build model. \n')
    # Build the model
    cnn_model = MODEL(
        include_top=False, input_shape=(width, width, 3), weights='imagenet', pooling='avg')
    inputs = Input((width, width, 3))
    x = inputs
    # x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, activation='elu', name='fc')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=inputs, outputs=x)

    return model
