from utils import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
pixel_mean = 84.53922408040364
pixel_std = 15.878809209259089

params = {'lr': 1e-4,
          'batch_size': 1,
          'epochs': 100,
          'n_feat_maps': 16,
          'seed': 2,
           }
dest_dir = 'results/target'
previous_dir = None

from numpy.random import seed
seed(params['seed'])
import tensorflow as tf
tf.random.set_seed(params['seed'])
import os
import numpy as np
import pickle
import random
from tensorflow.keras.backend import constant
from tensorflow.keras.models import load_model

print('params', params)
ct_train_image, ct_train_mask, ct_test_image, ct_test_mask, cbct_train_image, cbct_train_mask, cbct_test_image, cbct_test_mask = load_data()
history_keys = ['loss', 'loss_test']
history = {key: np.zeros((params['epochs'])) for key in history_keys}
save_dir = dest_dir + '/' + params2name(params)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
pickle.dump(params, open(save_dir + '/params.p', "wb"))
if previous_dir==None:
    print('Starting training from scratch')
    model = unet(params)
    epochs_previous = 0
else:
    file = open(save_dir + '/previous_dir.txt',"w")
    file.write(previous_dir)
    file.close()
    params_previous = pickle.load(open(previous_dir + '/params.p', "rb"))
    epochs_previous = params_previous['epochs']
    print('Starting training from a model trained for ' + str(epochs_previous) + ' epochs')
    model = load_model(previous_dir + '/weights.h5')
    history_previous = pickle.load(open(previous_dir + '/history.p', "rb"))
    for key in history_keys:
        history[key][0:epochs_previous] = history_previous[key]

# Initializations
x_batch = np.zeros((1,*image_size,1))
y_batch = np.zeros((1,*image_size,2))

for e in np.arange(epochs_previous, params['epochs']):

    cbct_inds = np.random.permutation(30)

    # Load transformed dataset
    cbct_train_image_transformed = np.load('/export/home/elbrion/unet_uda/bladder_160_160_128/base/data_augmented/cbct_train_image_transformed_' + str(e) + '.npy')
    cbct_train_mask_transformed = np.load('/export/home/elbrion/unet_uda/bladder_160_160_128/base/data_augmented/cbct_train_mask_transformed_' + str(e) + '.npy')

    # Training
    for batch_num in range(30):
        ind = cbct_inds[batch_num]
        x_batch[0,:,:,:,0] = cbct_train_image_transformed[ind,:,:,:,0]
        y_batch[0,:,:,:,:] = cbct_train_mask_transformed[ind,:,:,:,:]
        history_current = model.fit((x_batch-pixel_mean)/pixel_std, y_batch, epochs=1, batch_size=params['batch_size'], verbose=0)
        history['loss'][e] += history_current.history['loss'][0]
    history['loss'][e] /= 30

    for batch_num in range(30):
        x_batch[0, :, :, :, 0] = cbct_test_image[batch_num, :, :, :, 0]
        y_batch[0, :, :, :, :] = cbct_test_mask[batch_num, :, :, :, :]
        eval = model.evaluate((x_batch - pixel_mean) / pixel_std, y_batch,
                                    batch_size=params['batch_size'], verbose=0)
        history['loss_test'][e] += eval
    history['loss_test'][e] /= 30

    print('epoch: {} - loss: {:.2e} - loss_test: {:.2e}'.format(e, history['loss'][e], history['loss_test'][e]))

    if e==(params['epochs']-1):
        mean_dice_cbct_final = 0
        if not os.path.exists(dest_dir + '/' + params2name(params) + '/metrics'):
            os.makedirs(dest_dir + '/' + params2name(params) + '/metrics')
        for ind in range(30):
            x_batch[0, :, :, :, 0] = cbct_test_image[ind, :, :, :, 0]
            reference = cbct_test_mask[ind, :, :, :, 0]
            preds = model.predict([(x_batch - pixel_mean) / pixel_std])
            prediction_thr = (preds[0,:,:,:,0]>0.5)
            metrics = {'DSC': f1_score(reference.flatten(), prediction_thr.flatten())}
            pickle.dump(metrics, open(dest_dir + '/' + params2name(params) + '/metrics/cbct_test_' + str(ind) + '_metrics.p', "wb"))
            mean_dice_cbct_final += metrics['DSC']
        mean_dice_cbct_final /= 30
        history['mean_dice_cbct_final'] = mean_dice_cbct_final
        print('mean_dice_cbct_final', mean_dice_cbct_final)

pickle.dump( history, open( save_dir + '/history.p', "wb" ) )
model.save(save_dir + '/weights.h5')