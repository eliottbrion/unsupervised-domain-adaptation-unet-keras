import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Conv2D, MaxPooling2D, \
    Conv2DTranspose, Dropout, Flatten, Dense, Lambda

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from math import log10, floor
import matplotlib.pyplot as plt
from scipy import ndimage
import pickle
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist

from gradreverse import *

image_size = [80,80,64]

def b(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.math.divide(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)

def dice_loss(y_true, y_pred):
    return -b(y_true, y_pred)

def total_loss(segmentation_true, domain_true, segmentation_pred, domain_pred):
    return tf.add(dice_loss(segmentation_true, segmentation_pred), tensorflow.keras.losses.BinaryCrossentropy(domain_true, domain_pred))

def unet(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    pixel_mean = 86.62597169503347
    pixel_std = 15.921692390390666

    #inputs = (tf.cast(inputs, tf.float32) - pixel_mean) / pixel_std

    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(volumes)
    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck and feature extractor

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Upsampling

    copy4 = conv4
    up6 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), copy4],
      axis=4)
    conv6 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv6)

    copy3 = conv3
    up7 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy3],
      axis=4)
    print('up7.shape.dims', up7.shape.dims)
    conv7 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up7)
    print('conv7.shape.dims', conv7.shape.dims)
    conv7 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv7)
    print('conv7.shape.dims', conv7.shape.dims)

    copy2 = conv2
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy2],
      axis=4)
    print('up8.shape.dims', up8.shape.dims)
    conv8 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up8)
    print('conv8.shape.dims', conv8.shape.dims)
    conv8 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv8)
    print('conv8.shape.dims', conv8.shape.dims)

    copy1 = conv1
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy1],
      axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv9)

    outputs  = Conv3D(2, (1, 1, 1), activation='softmax')(conv9)

    model = Model(inputs=[volumes], outputs=[outputs])

    if params['optimizer']=='Adam' and params['loss']=='binary_crossentropy':
        model.compile(optimizer=Adam(params['lr']), loss='binary_crossentropy', metrics=[b])
    elif params['optimizer']=='Adam' and params['loss']=='dice_loss':
        model.compile(optimizer=Adam(params['lr']), loss=dice_loss)
    elif params['optimizer']=='SGD':
        print('SGD is being used.')
        model.compile(optimizer=SGD(params['lr'], momentum=0.9), loss='binary_crossentropy', metrics=[b])

    return model

def dann_L8(params):

    inputs = Input(batch_shape=(None, *image_size, 1))
    lambdaa = 0

    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(params['n_feat_maps'], (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck and feature extractor

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Upsampling

    copy4 = conv4
    up6 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), copy4],
        axis=4)
    conv6 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv6)

    copy3 = conv3
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy3],
        axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy2 = conv2
    up8 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy2],
        axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv8)
    model_feature = conv8
    conv8 = tf.slice(conv8, [0, 0, 0, 0, 0],[params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 2])

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0],[params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 1])
    up9 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy1],
        axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv9)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv9)

    # === Domain classifier ===

    feat = GradReverse(lambdaa)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu')(feat)
    convA = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu')(poolA)
    convB = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu')(convB)
    fl = Flatten()(convB)
    dense = Dense(params['d_arch'][4], activation='relu')(fl)

    domain_pred = Dense(1, activation = 'sigmoid', name='domain_output')(dense)

    model = Model(inputs=[inputs], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": 1.0}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics ={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model



