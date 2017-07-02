import numpy as np
import tensorflow as tf

from keras.layers import Convolution2D, BatchNormalization, Activation, merge, UpSampling2D, Lambda, Input
from keras.models import Model
from keras import backend as K


from sr.models import  conv_block, up_block, res_block

##
## RESTORE CNN
##

def restore_cnn_model(input_shape):
    inputs = Input(shape=input_shape)

    # 9-5-5 similar to SRCNN
    x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(inputs)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same', name='level2')(x)
    x = Convolution2D(3, (5, 5), activation='relu', padding='same', name='level3')(x)

    m = Model(inputs=inputs, outputs=x)
    return m

def restore_cnn_bn_model(input_shape):
    inputs = Input(shape=input_shape)

    # 9-5-5 w/ batch norm
    x = conv_block(inputs, 64, (9,9), activation='relu')
    x = conv_block(x, 32, (5,5), activation='relu')
    x = conv_block(x, 3, (5, 5), activation='relu')

    m = Model(inputs=inputs, outputs=x)
    return m


def restore_resnet_model(input_shape):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 64, (9,9))
    for i in range(4): x = res_block(x)
    x = Convolution2D(3, (9,9), activation='relu', padding='same')(x)

    m = Model(inputs=inputs, outputs=x)
    return m
