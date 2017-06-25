# -*- coding: utf-8 -*-
import keras.backend as K
import tensorflow as tf
import numpy as np

from datetime import datetime
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from models import create_espcnn_model, create_srcnn_model

import os
import errno
from scipy.misc import imread

##
## Images
##

def list_filenames(images_dir, full_path=True):
    file_names = [f for f in sorted(os.listdir(images_dir))
                  if (f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.jpg'))]
    if full_path:
        file_names = [os.path.join(images_dir, f) for f in file_names]
    return file_names

def get_filenames(lr_path="data/temp/lr/", hr_path="data/temp/hr/"):
    X_filenames = list_filenames(lr_path, full_path=True)
    Y_filenames = list_filenames(hr_path, full_path=True)
    return X_filenames, Y_filenames

def load_image(filename, mode="YCbCr"):
    img = imread(filename, mode=mode)
    return np.asarray(img, dtype=K.floatx()) / 255. # normalize

def get_images(filenames, mode="YCbCr"):
    return np.asarray([load_image(f, mode=mode) for f in filenames])

def get_input_shape(images_dir):
    filenames = list_filenames(images_dir)
    x = filenames[0]
    x = np.array(imread(x))
    return x.shape

##
## Iterator
##

def lr_hr_generator(lr_path, hr_path, mode='YCbCr'):
    X_filenames, Y_filenames = get_filenames(lr_path, hr_path)
    while 1:
        for x_file, y_file in zip(X_filenames, Y_filenames):
            x = load_image(x_file, mode=mode)
            y = load_image(y_file, mode=mode)
            
            x = np.reshape(x, (1,) + x.shape)
            y = np.reshape(y, (1,) + y.shape)   

            yield(x, y)
            
def steps_for_batch_size(images_dir, batch_size):
    X = list_filenames(images_dir)
    total = len(X)
    return max(1, int(total/batch_size))

##
## Loss
##

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    ref: https://github.com/titu1994/Image-Super-Resolution/blob/master/models.py
    """
    def log10(x):
        return K.log(x) / K.log(K.constant(10, dtype=K.floatx()))

    return -10. * log10(K.mean(K.square(y_pred - y_true)))

##
## util
##

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
 
class Pipeline():
    def __init__(self, input_root_dir, results_root_dir, network='espcnn'):
        self.root_dir = input_root_dir
        self.results_root_dir = results_root_dir
        self.network = network

        self.results_dir = self._get_and_prepare_results_dir()

    def _get_and_prepare_results_dir(self):
        """
            results_dir
                /model.h5
                /weights.h5
                /logs/
                    ...
        """
        # timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = "%s_%s" % (self.network, ts)

        # make output dirs
        results_dir = "%s/%s/" % (self.results_root_dir, model_name)
        mkdir_p(results_dir)
        mkdir_p(results_dir + 'logs/')

        print("\n\n[TRAIN]    saving results to %s\n" % results_dir)
        return results_dir

    def get_callbacks(self):
        # callbacks -- tensorboard
        log_dir = self.results_dir + 'logs/'
        tensorboard = TensorBoard(log_dir=log_dir)

        # callbacks -- model weights
        weights_path = self.results_dir + 'weights.h5' 
        model_checkpoint = ModelCheckpoint(monitor='loss', filepath=weights_path, save_best_only=True)

        # callbacks -- learning rate
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-5)
        return [tensorboard, model_checkpoint, reduce_lr]

    def run(self, scale=4, epochs=100, batch_size=32, save=True):  
        # input shape
        lr_path = self.root_dir + 'lr/'
        hr_path = self.root_dir + 'hr/'
        input_shape = get_input_shape(lr_path)

        # model
        if (self.network == 'srcnn'):
            model = create_srcnn_model(input_shape, scale=scale)
            model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=[PSNRLoss])            
        else:
            model = create_espcnn_model(input_shape, scale=scale)
            model.compile(loss='mse', optimizer=Adam(lr=1e-3), metrics=[PSNRLoss])

        # callbacks
        callbacks = self.get_callbacks()

        # train
        gen = lr_hr_generator(lr_path, hr_path)
        steps = steps_for_batch_size(lr_path, batch_size)
        model.fit_generator(gen, steps, epochs=epochs, callbacks=callbacks)

        # save
        if (save):
            model_path = self.results_dir + "model.h5"
            model.save(model_path)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Train SR model.")
    parser.add_argument("image_path", type=str, help="Path to input images, expects sub-directories /path/lr/ and /path/hr/.")
    parser.add_argument("--results_path", type=str, default="results/", help="Results base dir, will create subdirectories e.g. /results/model_timestamp/")
    parser.add_argument("--network", type=str, default="espcnn", help="Network architecture, [srcnn|espcnn]")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor. Default=4.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs. Default=100")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")

    args = parser.parse_args()

    image_path = args.image_path
    results_path = args.results_path
    network = args.network
    scale = args.scale
    epochs = args.epochs
    batch_size = args.batch_size

    # training pipeline
    p = Pipeline(image_path, results_path, network=network)
    p.run(scale=scale, epochs=epochs, batch_size=batch_size)
