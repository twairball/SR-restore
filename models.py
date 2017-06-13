import numpy as np
import tensorflow as tf

from keras.layers import Convolution2D, BatchNormalization, Activation, merge, UpSampling2D, Lambda, Input
from keras.models import Model
from keras import backend as K


##
## SR CNN
##
def create_srcnn_model(input_shape, scale=3):

    inputs = Input(shape=input_shape)

    # 9-5-5 (see paper)
    x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(inputs)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same', name='level2')(x)
    x = Convolution2D(3, (5, 5), activation='relu', padding='same', name='level3')(x)

    m = Model(inputs=inputs, outputs=x)
    return m


##
## Subpixel ESPCN
##
def create_espcnn_model(input_shape, scale=3):
    inputs = Input(shape=input_shape)
    channels = input_shape[-1] # TF channel-last

    # 5-3-3 (see paper)
    x = Convolution2D(64, (5, 5), activation='tanh', padding='same', name='level1')(inputs)
    x = Convolution2D(32, (3, 3), activation='tanh', padding='same', name='level2')(x)
    x = Convolution2D(channels * scale ** 2, (3, 3), activation='tanh', padding='same', name='level3')(x)

    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    out = Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')(x)

    m = Model(inputs=inputs, outputs=out)
    return m



##
## Perceptive loss - SR blocks
##
def conv_block(x, filters, size, stride=(2,2), mode='same', act=True):
    x = Convolution2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x) if act else x

def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1,1))
    x = conv_block(x, nf, 3, (1,1), act=False)
    return merge([x, ip], mode='sum')

def up_block(x, filters, size):
    x = UpSampling2D()(x)
    x = Convolution2D(filters, size, size, border_mode='same')(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x)
