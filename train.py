# -*- coding: utf-8 -*-
import keras.backend as K
import tensorflow as tf
import numpy as np

from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from models import create_espcnn_model, create_srcnn_model

import os
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

def get_images(filenames):
    return np.array([imread(f, mode='YCbCr') for f in filenames])

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
            x = imread(x_file, mode=mode)
            y = imread(y_file, mode=mode)
            
            x = np.reshape(x, (1,) + x.shape)
            y = np.reshape(y, (1,) + y.shape)

            yield(x, y)
            
def steps_for_batch_size(images_dir, batch_size):
    X, _ = get_filenames(images_dir, images_dir)
    total = len(X)
    return max(1, int(total/batch_size))


##
## PSNR -- pixel loss
##

def log10(x):
    """
    tensorflowにはlog10がないので自分で定義
    https://github.com/tensorflow/tensorflow/issues/1666
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


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
    # tensorflowオブジェクトを返さないといけないので以下ではNG
    # return -10. * np.log10(K.mean(K.square(y_pred - y_true)))
    return -10. * log10(K.mean(K.square(y_pred - y_true)))


def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))


def experiment(
    root_dir='data/temp/', 
    models_dir='results/',
    logs_dir='results/logs/', 
    weights_dir='results/weights/', 
    scale=4, epochs=100, batch_size=32):

    # input shape
    lr_path = root_dir + 'lr/'
    hr_path = root_dir + 'hr/'
    input_shape = get_input_shape(lr_path)

    # timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # callbacks -- tensorboard
    log_dir = logs_dir + ts
    tensorboard = TensorBoard(log_dir=log_dir)

    # callbacks -- model weights
    weights_path = weights_dir + ("espcnn_weights_%s.h5" % ts)
    model_checkpoint = ModelCheckpoint(monitor='loss', filepath=weights_path, save_best_only=True)

    # callbacks -- learning rate
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.0001)

    # model
    model = create_espcnn_model(input_shape, scale=scale)
    model.compile(loss=PSNRLoss, optimizer='rmsprop', metrics=[PSNRLoss])

    # train
    gen = lr_hr_generator(lr_path, hr_path)
    steps = steps_for_batch_size(lr_path, batch_size)
    model.fit_generator(gen, steps, epochs=epochs, callbacks=[tensorboard, reduce_lr, model_checkpoint])

    # save model
    model_path = models_dir + ("espcnn_%s.h5" % ts)
    model.save(model_path)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Train SR model.")
    parser.add_argument("image_path", type=str, help="Path to input images, expects sub-directories /path/lr/ and /path/hr/.")
    parser.add_argument("--logs", type=str, default='results/logs/', help="Logs output path.")
    parser.add_argument("--weights", type=str, default='results/weights/', help="Weights output path.")
    parser.add_argument("--save", type=str, default='results/', help="Model save path.")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor. Default=4.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs. Default=100")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Default=32")

    args = parser.parse_args()

    image_path = args.image_path
    models_path = args.save
    logs_path = args.logs
    weights_path = args.weights
    scale = args.scale
    epochs = args.epochs
    batch_size = args.batch_size

    # run experiment
    experiment(image_path,
               models_dir=models_path,
               logs_dir=logs_path,
               weights_dir=weights_path,
               scale=scale,
               epochs=epochs,
               batch_size=batch_size)
