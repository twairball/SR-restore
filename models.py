import numpy as np
import tensorflow as tf

from keras.layers import Convolution2D, BatchNormalization, Activation, merge, UpSampling2D, Lambda, Input
from keras.models import Model
from keras import backend as K


from subpixel import SubpixelConv2D



##
## SR CNN
##
def create_srcnn_model(input_shape, scale=4):
    inputs = Input(shape=input_shape)
    channels = input_shape[-1] # TF channel-last

    # 9-5-5 (see paper)
    x = UpSampling2D((scale, scale))(inputs) # upsample
    x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(x)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same', name='level2')(x)
    x = Convolution2D(3, (5, 5), activation='relu', padding='same', name='level3')(x)

    m = Model(inputs=inputs, outputs=x)
    return m


##
## Subpixel ESPCN
##
def create_espcnn_model(input_shape, scale=4):
    inputs = Input(shape=input_shape)
    channels = input_shape[-1] # TF channel-last

    # 5-3-3 (see paper)
    x = Convolution2D(64, (5, 5), activation='relu', padding='same', name='level1')(inputs)
    x = Convolution2D(32, (3, 3), activation='relu', padding='same', name='level2')(x)
    x = Convolution2D(channels * scale ** 2, (3, 3), activation='relu', padding='same', name='level3')(x)

    # upsample
    out = SubpixelConv2D(input_shape, scale=scale)(x)

    m = Model(inputs=inputs, outputs=out)
    return m

def create_espcnn_bn_model(input_shape, scale=4):
    inputs = Input(shape=input_shape)
    channels = input_shape[-1] # TF channel-last

    # 5-3-3 (see paper)
    x = Convolution2D(64, (5, 5), padding='same', name='level1')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(32, (3, 3), padding='same', name='level2')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = Convolution2D(channels * scale ** 2, (3, 3), padding='same', name='level3')(x)

    # upsample
    out = SubpixelConv2D(input_shape, scale=scale)(x)

    m = Model(inputs=inputs, outputs=out)
    return m

##
## SR filter blocks
##
def conv_block(inputs, filters, kernel_size, strides=(1,1), padding="same", activation='relu'):
    x = Convolution2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def up_block(inputs, filters, kernel_size, strides=(1,1), scale=2, padding="same", activation="relu"):
    size = (scale,scale)
    x = UpSampling2D(size)(inputs)
    x = Convolution2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def res_block(inputs, filters=64):
    x = conv_block(inputs, filters, (3,3))
    x = conv_block(x, filters, (3,3), activation=False)
    return merge([x, inputs], mode='sum')


##
## Resnet batchnorm w/ NN upsampling
##

def create_resnet_up_model(input_shape, scale=4):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 64, (9,9))
    for i in range(4): x = res_block(x)
    x = up_block(x, 64, (3, 3), scale=scale / 2)
    x = up_block(x, 64, (3, 3), scale=scale / 2)
    out = Convolution2D(3, (9,9), activation='relu', padding='same')(x)

    m = Model(inputs=inputs, outputs=out)
    return m


