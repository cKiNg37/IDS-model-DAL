import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from data_processing import load_data
import  sklearn as sk
import pandas as pd
import numpy as np
from numpy import concatenate
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU,AlphaDropout,BatchNormalization,Conv1D
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from keras.regularizers import l2
from keras import backend as K
from  AttentionLayer import AttentionLayer
from tensorflow.keras.models import load_model


def DenseLayer(x, cn_filter, drop_rate, ax,i):
    # Bottleneck layers
    if i:
      x = BatchNormalization(axis=ax)(x)
      x = Conv1D(cn_filter,1,activation = 'elu',dilation_rate = 5)(x)
    # Composite function
      x = BatchNormalization(axis=ax)(x)
      x = Conv1D(cn_filter, 3,activation = 'elu',dilation_rate = 7)(x)
    else:
    # Bottleneck layers
        x = BatchNormalization(axis=ax)(x)
        x = Conv1D(cn_filter, 1, activation='elu', dilation_rate=2)(x)
    # Composite function
        x = BatchNormalization(axis=ax)(x)
        x = Conv1D(cn_filter, 3, activation='elu', dilation_rate=3)(x)

    if drop_rate: x = AlphaDropout(drop_rate)(x)
    return x


def DenseBlock(x, cn_layers, filter, drop_rate, ax):
    for ii in range(cn_layers):
        conv = DenseLayer(x, cn_filter=filter, drop_rate=drop_rate, ax=ax, i=ii)
        # print('1',conv)
        x = layers.Lambda(lambda x:K.concatenate([x, conv],axis=ax))(x)
        # x = K.concatenate([x, conv],axis=1)
    return x


def TransitionLayer(x, ax):
    x = BatchNormalization(axis=ax)(x)
    x = layers.GlobalMaxPooling1D()(x)
    return x





def DenseLayer_multi(x, cn_filter, drop_rate, ax):
    # Bottleneck layers
    # if i:
    x = BatchNormalization(axis=ax)(x)
    x = Conv1D(cn_filter,3,activation = 'elu')(x)
    # Composite function
    x = BatchNormalization(axis=ax)(x)
    x = Conv1D(cn_filter, 3,activation = 'elu')(x)
    # else:
    # # Bottleneck layers
    #   x = BatchNormalization(axis=ax)(x)
    #   x = Conv1D(cn_filter, 1, activation='elu', dilation_rate=2)(x)
    # # Composite function
    #   x = BatchNormalization(axis=ax)(x)
    #   x = Conv1D(cn_filter, 3, activation='elu', dilation_rate=3)(x)

    if drop_rate: x = AlphaDropout(drop_rate)(x)
    return x


def DenseBlock_multi(x, cn_layers, filter, drop_rate, ax):
    for ii in range(cn_layers):
        conv = DenseLayer_multi(x, cn_filter=filter, drop_rate=drop_rate, ax=ax)
        # print('1',conv)
        x = layers.Lambda(lambda x:K.concatenate([x, conv],axis=ax))(x)
        # x = K.concatenate([x, conv],axis=1)
    return x


def TransitionLayer_multi(x, ax):
    x = BatchNormalization(axis=ax)(x)
    x = layers.GlobalMaxPooling1D()(x)
    return x