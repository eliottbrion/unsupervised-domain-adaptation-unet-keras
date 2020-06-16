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

image_size = [160,160,128]

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

def unet(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck and feature extractor

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5],
        axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = conv4
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4],
      axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = conv3
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3],
      axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = conv2
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2],
      axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = conv1
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1],
      axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    outputs  = Conv3D(2, (1, 1, 1), activation='softmax')(conv11)

    model = Model(inputs=[volumes], outputs=[outputs])

    model.compile(optimizer=Adam(params['lr']), loss=dice_loss)

    return model

def unet_L6(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)
    model_feature = conv6
    conv6 = tf.slice(conv6, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 5, 5, 4, params['n_feat_maps'] * 32])

    # Upsampling

    copy5 = tf.slice(conv5, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 10, 10, 8, params['n_feat_maps'] * 16])
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5], axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = tf.slice(conv4, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 20, 20, 16, params['n_feat_maps'] * 8])
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4], axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = tf.slice(conv3, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 4])
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3], axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = tf.slice(conv2, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2], axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1], axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    fl = Flatten()(convA)
    dense = Dense(params['d_arch'][1], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model

def unet_L7(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5], axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)
    model_feature = conv7
    conv7 = tf.slice(conv7, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 10, 10, 8, params['n_feat_maps'] * 16])

    copy4 = tf.slice(conv4, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 20, 20, 16, params['n_feat_maps'] * 8])
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4], axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = tf.slice(conv3, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 4])
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3], axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = tf.slice(conv2, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2], axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1], axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(poolA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(convB)
    fl = Flatten()(convB)
    dense = Dense(params['d_arch'][2], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model

def unet_L8(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5], axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = conv4
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4], axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)
    model_feature = conv8
    conv8 = tf.slice(conv8, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 20, 20, 16, params['n_feat_maps'] * 8])

    copy3 = tf.slice(conv3, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 4])
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3], axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)


    copy2 = tf.slice(conv2, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2], axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1], axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(poolA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(convB)
    poolB = MaxPooling3D((2, 2, 2))(convB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(poolB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(convC)
    fl = Flatten()(convC)
    dense = Dense(params['d_arch'][3], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model

def unet_L9(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5],
        axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = conv4
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4],
      axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = conv3
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3],
      axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)
    model_feature = conv9
    conv9 = tf.slice(conv9, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 40, 40, 32, params['n_feat_maps'] * 4])

    copy2 = tf.slice(conv2, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2],
      axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1],
      axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(poolA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(convB)
    poolB = MaxPooling3D((2, 2, 2))(convB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(poolB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(convC)
    poolC = MaxPooling3D((2, 2, 2))(convC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(poolC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(convD)
    fl = Flatten()(convD)
    dense = Dense(params['d_arch'][3], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model

def unet_L10(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5],
        axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = conv4
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4],
      axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = conv3
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3],
      axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = conv2
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2],
      axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)
    model_feature = conv10
    conv10 = tf.slice(conv10, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 80, 80, 64, params['n_feat_maps'] * 2])

    copy1 = tf.slice(conv1, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1],
      axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(poolA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(convB)
    poolB = MaxPooling3D((2, 2, 2))(convB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(poolB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(convC)
    poolC = MaxPooling3D((2, 2, 2))(convC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(poolC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(convD)
    poolD = MaxPooling3D((2, 2, 2))(convD)
    convE = Conv3D(params['d_arch'][4], (3, 3, 3), activation='relu', padding='same')(poolD)
    convE = Conv3D(params['d_arch'][4], (3, 3, 3), activation='relu', padding='same')(convE)
    fl = Flatten()(convE)
    dense = Dense(params['d_arch'][5], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model

def unet_L11(params):

    volumes = Input(batch_shape=(None, *image_size, 1))

    # Downsampling

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

    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    # Bottleneck

    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(pool5)
    conv6 = Conv3D(params['n_feat_maps'] * 32, (3, 3, 3), activation='relu', padding='same')(conv6)

    # Upsampling

    copy5 = conv5
    up7 = concatenate(
        [Conv3DTranspose(params['n_feat_maps'] * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), copy5],
        axis=4)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(params['n_feat_maps'] * 16, (3, 3, 3), activation='relu', padding='same')(conv7)

    copy4 = conv4
    up8 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), copy4],
      axis=4)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(params['n_feat_maps'] * 8, (3, 3, 3), activation='relu', padding='same')(conv8)

    copy3 = conv3
    up9 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), copy3],
      axis=4)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(params['n_feat_maps'] * 4, (3, 3, 3), activation='relu', padding='same')(conv9)

    copy2 = conv2
    up10 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv9), copy2],
      axis=4)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(up10)
    conv10 = Conv3D(params['n_feat_maps'] * 2, (3, 3, 3), activation='relu', padding='same')(conv10)

    copy1 = conv1
    up11 = concatenate(
      [Conv3DTranspose(params['n_feat_maps'] * 1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv10), copy1],
      axis=4)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(up11)
    conv11 = Conv3D(params['n_feat_maps'] * 1, (3, 3, 3), activation='relu', padding='same')(conv11)
    model_feature = conv11
    conv11 = tf.slice(conv11, [0, 0, 0, 0, 0], [params['batch_size'] // 2, 160, 160, 128, params['n_feat_maps'] * 1])

    segmentation_pred = Conv3D(2, (1, 1, 1), activation = 'softmax', name="segmentation_output")(conv11)

    # === Domain classifier ===

    feat = GradReverse(0.)(model_feature)

    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(feat)
    convA = Conv3D(params['d_arch'][0], (3, 3, 3), activation='relu', padding='same')(convA)
    poolA = MaxPooling3D((2, 2, 2))(convA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(poolA)
    convB = Conv3D(params['d_arch'][1], (3, 3, 3), activation='relu', padding='same')(convB)
    poolB = MaxPooling3D((2, 2, 2))(convB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(poolB)
    convC = Conv3D(params['d_arch'][2], (3, 3, 3), activation='relu', padding='same')(convC)
    poolC = MaxPooling3D((2, 2, 2))(convC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(poolC)
    convD = Conv3D(params['d_arch'][3], (3, 3, 3), activation='relu', padding='same')(convD)
    poolD = MaxPooling3D((2, 2, 2))(convD)
    convE = Conv3D(params['d_arch'][4], (3, 3, 3), activation='relu', padding='same')(poolD)
    convE = Conv3D(params['d_arch'][4], (3, 3, 3), activation='relu', padding='same')(convE)
    poolE = MaxPooling3D((2, 2, 2))(convE)
    convF = Conv3D(params['d_arch'][5], (3, 3, 3), activation='relu', padding='same')(poolE)
    convF = Conv3D(params['d_arch'][5], (3, 3, 3), activation='relu', padding='same')(convF)
    fl = Flatten()(convF)
    dense = Dense(params['d_arch'][6], activation='relu')(fl)

    domain_pred = Dense(1, activation='sigmoid', name='domain_output')(dense)

    model = Model(inputs=[volumes], outputs=[segmentation_pred, domain_pred])

    losses = {
        "segmentation_output": dice_loss,
        "domain_output": "binary_crossentropy",
    }
    lossWeights = {"segmentation_output": 1.0, "domain_output": params['psi']}

    model.compile(optimizer=Adam(params['lr']), loss=losses, loss_weights=lossWeights,
                  metrics={'segmentation_output': None, 'domain_output': 'accuracy'})

    return model







