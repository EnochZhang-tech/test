from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Add,multiply,concatenate,Conv1D,Concatenate
from keras.layers.core import  *
from keras.models import *
import keras.backend as K


#%%
def cbam_block(cbam_feature, ratio=5):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
    cbam_feature = channel_attention(cbam_feature, ratio)
    # cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


# def channel_attention(input_feature,ratio=5):
#     # channel=input_feature._keras_shape[-1]
#     channel = K.int_shape(input_feature)[-1]
#     shared_layer_one = Dense(channel // ratio,
#                              activation='tanh',
#                              kernel_initializer='he_normal',
#                              # He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入
#                              use_bias=True,
#                              bias_initializer='zeros')
#     shared_layer_two = Dense(channel,
#                              kernel_initializer='he_normal',
#                              use_bias=True,
#                              bias_initializer='zeros')
#
#     avg_pool = GlobalAveragePooling1D()(input_feature)
#     avg_pool = Reshape((1, channel))(avg_pool)
#     assert avg_pool._keras_shape[1:] == (1, channel)
#     avg_pool = shared_layer_one(avg_pool)
#     assert avg_pool._keras_shape[1:] == (1, channel // ratio)
#     avg_pool = shared_layer_two(avg_pool)
#     assert avg_pool._keras_shape[1:] == (1, channel)
#
#     max_pool = GlobalMaxPooling1D()(input_feature)
#     max_pool = Reshape((1, channel))(max_pool)
#     assert max_pool._keras_shape[1:] == (1, channel)
#     max_pool = shared_layer_one(max_pool)
#     assert max_pool._keras_shape[1:] == (1, channel // ratio)
#     max_pool = shared_layer_two(max_pool)
#     assert max_pool._keras_shape[1:] == (1, channel)
#
#
#     cbam_feature = Add()([avg_pool, max_pool])
#     cbam_feature = Activation('sigmoid')(cbam_feature)
#
#
#     return multiply([input_feature, cbam_feature])
#
# def spatial_attention(input_feature):
#     kernel_size = 7
#     cbam_feature=input_feature
#     avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(cbam_feature)
#     assert avg_pool._keras_shape[-1] == 1
#     max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(cbam_feature)
#     assert max_pool._keras_shape[-1] == 1
#     concat = Concatenate(axis=2)([avg_pool, max_pool])
#     assert concat._keras_shape[-1] == 2
#     cbam_feature = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
#                           activation='tanh',
#                           kernel_initializer='he_normal',
#                           use_bias=False)(concat)
#
#     assert cbam_feature._keras_shape[-1] == 1
#
#
#     return multiply([input_feature, cbam_feature])

import keras.backend as K
from keras.layers import Dense, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Lambda, Concatenate, Conv1D, Add, \
    Activation, multiply


def channel_attention(input_feature, ratio=5):
    channel = K.int_shape(input_feature)[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='tanh',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(input_feature)

    concat = Concatenate(axis=2)([avg_pool, max_pool])

    cbam_feature = Conv1D(filters=1, kernel_size=kernel_size, strides=1, padding='same',
                          activation='tanh',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return multiply([input_feature, cbam_feature])
