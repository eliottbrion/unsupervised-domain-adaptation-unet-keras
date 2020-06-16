from utils import *
from gradreverse import *
from models import *

pixel_mean = 84.53922408040364
pixel_std = 15.878809209259089
image_size = [160,160,128]

def train(dest_dir, previous_dir, params):

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
    history_keys = ['sl', 'sl_test', 'dl', 'dl_test', 'loss', 'loss_test', 'lambda', 'd_acc', 'd_acc_test',
                    'dice_loss_cbct_test', 'dice_cbct_test']
    history = {key: np.zeros((params['epochs'])) for key in history_keys}
    save_dir = dest_dir + '/' + params2name(params)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pickle.dump(params, open(save_dir + '/params.p', "wb"))
    if previous_dir==None:
        print('Starting training from scratch')
        if params['model']=='unet_L8':
            model = unet_L8(params)
        elif params['model']=='unet_L9':
            model = unet_L9(params)
        elif params['model']=='unet_L10':
            model = unet_L10(params)
        elif params['model']=='unet_L11':
            model = unet_L11(params)
        epochs_previous = 0
    else:
        file = open(save_dir + '/previous_dir.txt',"w")
        file.write(previous_dir)
        file.close()
        params_previous = pickle.load(open(previous_dir + '/params.p', "rb"))
        epochs_previous = params_previous['epochs']
        print('Starting training from a model trained for ' + str(epochs_previous) + ' epochs')
        model = load_model(previous_dir + '/weights.h5', custom_objects={'dice_loss': dice_loss, 'GradReverse': GradReverse})
        history_previous = pickle.load(open(previous_dir + '/history.p', "rb"))
        for key in history_keys:
            history[key][0:epochs_previous] = history_previous[key]

    # Initializations
    x_batch = np.zeros((2,*image_size,1))
    y_batch = np.zeros((1,*image_size,2))

    for e in np.arange(epochs_previous, params['epochs']):
        history['lambda'][e] = np.max([0,params['l_max']*(e-params['e1'])/(params['epochs']-params['e1'])])
        model.get_layer('grad_reverse').l.assign(history['lambda'][e]/params['psi'])
        ct_inds = np.random.permutation(30)
        cbct_inds = np.random.permutation(30)

        # Load transformed dataset
        ct_train_image_transformed = np.load('data_augmented/ct_train_image_transformed_' + str(e) + '.npy')
        ct_train_mask_transformed = np.load('data_augmented/ct_train_mask_transformed_' + str(e) + '.npy')
        cbct_train_image_transformed = np.load('data_augmented/cbct_train_image_transformed_' + str(e) + '.npy')

        # Training
        for batch_num in range(30):
            ind = ct_inds[batch_num]
            x_batch[0,:,:,:,0] =ct_train_image_transformed[ind,:,:,:,0]
            y_batch[0,:,:,:,:] = ct_train_mask_transformed[ind,:,:,:,:]
            ind = cbct_inds[batch_num]
            x_batch[1, :, :, :, 0] = cbct_train_image_transformed[ind,:,:,:,0]
            loss_batch, sl_batch, dl_batch, d_acc_batch = model.train_on_batch([(x_batch-pixel_mean)/pixel_std],
                                            {'segmentation_output': y_batch, 'domain_output': np.asarray([0.,1.])})
            history['loss'][e] += loss_batch
            history['sl'][e] += sl_batch
            history['dl'][e] += dl_batch
            history['d_acc'][e] += d_acc_batch
        for key in ['loss', 'sl', 'dl', 'd_acc']:
            history[key][e] /= 30

        # Evaluation
        for ind in range(30):
            x_batch[0,:,:,:,0] = ct_test_image[ind,:,:,:,0]
            x_batch[1, :, :, :, 0] = cbct_test_image[ind, :, :, :, 0]
            y_batch[0, :, :, :, :] = ct_test_mask[ind, :, :, :, :]
            loss_batch, sl_batch, dl_batch, d_acc_batch = model.test_on_batch([(x_batch-pixel_mean)/pixel_std],
                                        {'segmentation_output': y_batch, 'domain_output': np.asarray([0.,1.])})
            history['loss_test'][e] += loss_batch
            history['sl_test'][e] += sl_batch
            history['dl_test'][e] += dl_batch
            history['d_acc_test'][e] += d_acc_batch
        for key in ['loss_test', 'sl_test', 'dl_test', 'd_acc_test']:
            history[key][e] /= 30

        for ind in range(30):
            x_batch[0,:,:,:,0] = cbct_test_image[ind,:,:,:,0]
            y_batch[0, :, :, :, :] = cbct_test_mask[ind, :, :, :, :]
            _, dice_cbct_batch, _, _ = model.test_on_batch([(x_batch-pixel_mean)/pixel_std],
                                        {'segmentation_output': y_batch, 'domain_output': np.asarray([0.,1.])})
            history['dice_loss_cbct_test'][e] += dice_cbct_batch
        history['dice_loss_cbct_test'][e] /= 30

        print('epoch: {} - d_acc: {:.2e} - d_acc_test: {:.2e} - sl: {:.2e} - dice_loss_cbct_test: {:.2e} - l: {:.2e}'.format(e,history['d_acc'][e],history['d_acc_test'][e],history['sl'][e],history['dice_loss_cbct_test'][e],history['lambda'][e]))

        if e==(params['epochs']-1):
            mean_dice_cbct_final = 0
            if not os.path.exists(dest_dir + '/' + params2name(params) + '/metrics'):
                os.makedirs(dest_dir + '/' + params2name(params) + '/metrics')
            for ind in range(30):
                x_batch[0, :, :, :, 0] = cbct_test_image[ind, :, :, :, 0]
                reference = cbct_test_mask[ind, :, :, :, 0]
                preds = model.predict([(x_batch - pixel_mean) / pixel_std])
                preds = np.asarray(preds)
                prediction_thr = (preds[0][0,:,:,:,0]>0.5)
                metrics = {'DSC': f1_score(reference.flatten(), prediction_thr.flatten())}
                pickle.dump(metrics, open(dest_dir + '/' + params2name(params) + '/metrics/cbct_test_' + str(ind) + '_metrics.p', "wb"))
                mean_dice_cbct_final += metrics['DSC']
            mean_dice_cbct_final /= 30
            history['mean_dice_cbct_final'] = mean_dice_cbct_final
            print('mean_dice_cbct_final', mean_dice_cbct_final)

    pickle.dump( history, open( save_dir + '/history.p', "wb" ) )
    model.save(save_dir + '/weights.h5')