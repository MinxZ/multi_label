import numpy as np
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
from keras.layers import *
from keras.models import *

x_test = np.load('../data/X_test.npy')
width = x_test.shape[1]
model_name = "Xception"
MODEL = Xception

cnn_model = MODEL(
    include_top=False, input_shape=(width, width, 3), weights=None, pooling='avg')
inputs = Input((width, width, 3))
x = inputs
x = Lambda(preprocess_input, name='preprocessing')(x)
x = cnn_model(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='elu', name='fc')(x)
# x = Dropout(0.5)(x)
# x = Dense(n_class, activation='sigmoid', name='predictions')(x)
model = Model(inputs=inputs, outputs=x)
model.load_weights(f'../models/{model_name}.h5', by_name=True)


batch_size = 64
for train_test in ['train', 'test']:
    X = np.load(f'../data/X_{train_test}.npy')
    features = model.predict(X, batch_size=batch_size, verbose=1)
    np.save(f'../data/features256_{model_name}_{train_test}', features)


# model.load_weights(f'../models/Xception_69_256.h5', by_name=True)
#
# batch_size = 64s
# for train_test in ['train', 'val']:
#     X = np.load(f'../data/x_{train_test}.npy')
#     features = model.predict(X, batch_size=batch_size, verbose=1)
#     np.save(f'../data/features3_{model_name}_{train_test}', features)


def tri_128(fc_1, fc_2, fc_3):
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
    model = tri_fc(x, fc_1, fc_2, fc_3)

    x = concatenate([processed_a, processed_b, processed_c])

    model = Model(inputs, x)

    model.load_weights('../models/Xception_69_256.h5', by_name=True)
    model.load_weights(f'../models/tri.h5', by_name=True)
    # model.load_weights('../models/Xception_tri.h5', by_name=True)

    batch_size = 64
    for train_test in ['train', 'test']:
        X = np.load(f'../data/X_{train_test}.npy')
        features = model.predict(X, batch_size=batch_size, verbose=1)
        np.save(f'../data/Xception_tri_384_{train_test}', features)
