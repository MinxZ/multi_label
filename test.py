from __future__ import absolute_import, division, print_function

import argparse
import multiprocessing as mp
import random

import numpy as np
from keras import backend
from keras.applications import *
from keras.layers import *
from keras.models import *
from keras.utils.generic_utils import CustomObjectScope
from tqdm import tqdm

from image import *
from model import *


def run(model_name):
    # Load datasets
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(model='test')

    def predict():
        val_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input)
        test_result = model.evaluate_generator(
            val_datagen.flow(x_test, y_test, batch_size=batch_size),
            steps=len(x_test) / batch_size,
            max_queue_size=64,
            workers=16,
            use_multiprocessing=True,
            verbose=1)

        with open(f"../models/test_score.txt", "a") as text_file:
            text_file.write(f'{weight_name}: {str(test_result)}\n')
        return test_result

    if model_name == "ResNet50":
        print('\n  For Resnet')
        from keras.applications.imagenet_utils import preprocess_input
    elif model_name[:-3] == "DenseNet":
        print('\n  For DenseNet')
        from keras.applications.densenet import preprocess_input
    else:
        print('\n  For model = tf')
        from keras.applications.inception_v3 import preprocess_input

    # Loading model
    print('\n  Loading model')
    model_config, fc, pred, layer_names, input_shape = load_model_config()
    batch_size = model_config[model_name][0]

    weight_name = f'{model_name}_{len(fc)}_fc'
    for weight_name in [f'{model_name}_{len(fc)}_fc', f'{model_name}_{len(fc)}_fc_fine_tune']:
        with CustomObjectScope({'f1_loss': f1_loss, 'f1_score': f1_score}):
            model = load_model(f'../models/{weight_name}.h5')
        print('\n  Ready to test.')
        test_result = predict()

            model.compile(optimizer=opt, loss=losses,
                          loss_weights=lossWeights, metrics=metrics)

    quit()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hyper parameter")
    parser.add_argument(
        "--model", help="Model to use", default="All", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model)
