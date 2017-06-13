import keras.backend as K
import tensorflow as tf
import numpy as np

from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from models import create_espcnn_model, create_srcnn_model

import os
from scipy.misc import imread

##
## Images
##

def get_filenames(lr_path="data/temp/lr/", hr_path="data/temp/hr/"):
    file_names = [f for f in sorted(os.listdir(lr_path))
                  if (f.endswith('.jpeg') or f.endswith('.png') or f.endswith('.jpg'))]
    X_filenames = [os.path.join(lr_path, f) for f in file_names]
    Y_filenames = [os.path.join(hr_path, f) for f in file_names]
    return X_filenames, Y_filenames

def get_images(filenames):
    return np.array([imread(f, mode='YCbCr') for f in filenames])

##
## Utils
##
def preprocess_vgg(input):
    rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    preproc = lambda x: (x - rn_mean)[:, :, :, ::-1]
    return preproc


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


def experiment(root_dir='data/temp/', logs_dir='results/logs/', checkpoints_dir='results/', scale=4):
    # read data
    X_filenames, Y_filenames = get_filenames(root_dir + 'lr/', root_dir + 'hr/')
    X = get_images(X_filenames)
    Y = get_images(Y_filenames)
    input_shape = X[0].shape

    # callbacks -- tensorboard
    log_dir = (logs_dir + "{}").format(time())
    tensorboard = TensorBoard(log_dir=log_dir)

    # callbacks -- model checkpoints
    model_path = (checkpoints_dir + "espcnn_{}.h5").format(time())
    model_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True)

    # callbacks -- learning rate
    reduce_lr = ReduceLROnPlateau(monitor='PSNRLoss', factor=0.1, patience=5, min_lr=0.0001)

    # model
    espcnn = create_espcnn_model(input_shape, scale=scale)
    espcnn.compile(loss=PSNRLoss, optimizer='rmsprop', metrics=[PSNRLoss])

    # train
    batch_size = min(32, len(X))
    espcnn.fit(X, Y, batch_size=batch_size, epochs=100, callbacks=[tensorboard, reduce_lr, model_checkpoint])


if __name__ == '__main__':
    # run experiment
    experiment('data/temp/', 'results/logs/', 'results/', scale=4)
