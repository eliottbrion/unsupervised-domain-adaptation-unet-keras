import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

image_size = [160,160,128]

def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) +'_'
    results_name = results_name[:-1]
    return results_name

def plot_results(dest_dir, params):
    filepath = dest_dir + '/' + params2name(params) + '/history.p'
    history = pickle.load( open( filepath, "rb" ) )
    plt.figure(figsize=(20, 10))
    #plt.suptitle(params2name(params) + '\n \n' + 'dice_cbct_test=%.3f' % history['mean_dice_cbct_final'])
    plt.suptitle(params2name(params))
    plt.subplot(2, 3, 1)
    plt.plot(history['sl'], color='#1f77b4', label='segmentation loss train')
    plt.plot(history['sl_test'], color='#ff7f0e', label='segmentation loss test')
    plt.plot(history['dl'], color='#1f77b4', linestyle='--', label='domain loss train')
    plt.plot(history['dl_test'], color='#ff7f0e', linestyle='--', label='domain loss test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.subplot(2, 3, 2)
    plt.plot(history['loss'])
    plt.plot(history['loss_test'])
    plt.xlabel('Epoch')
    plt.ylabel('Total loss')
    plt.grid()
    plt.subplot(2, 3, 3)
    plt.plot(history['lambda'])
    plt.xlabel('Epoch')
    plt.ylabel('Lambda')
    plt.grid()
    plt.subplot(2, 3, 4)
    plt.plot(history['d_acc'])
    plt.plot(history['d_acc_test'])
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator accuracy')
    plt.grid()
    plt.subplot(2, 3, 5)
    plt.plot(history['dice_loss_cbct_test'])
    plt.xlabel('Epoch')
    plt.ylabel('CBCT segmentation loss test')
    plt.grid()
    filepath = dest_dir + '/' + params2name(params) + '/curves_' + params2name(params) + '.png'
    plt.savefig(filepath)

def image_transform(image, shears, angles, shifts, order):
    shear_matrix = np.array([[1, shears[0], shears[1], 0],
                         [shears[2], 1, shears[3], 0],
                         [shears[4], shears[5], 1, 0],
                         [0, 0, 0, 1]])

    shift_matrix = np.array([[1, 0, 0, shifts[0]],
                         [0, 1, 0, shifts[1]],
                         [0, 0, 1, shifts[2]],
                         [0, 0, 0, 1]])

    offset = np.array([[1, 0, 0, int(image_size[0]/2)],
                   [0, 1, 0, int(image_size[1]/2)],
                   [0, 0, 1, int(image_size[2]/2)],
                   [0, 0, 0, 1]])

    offset_opp = np.array([[1, 0, 0, -int(image_size[0]/2)],
                   [0, 1, 0, -int(image_size[1]/2)],
                   [0, 0, 1, -int(image_size[2]/2)],
                   [0, 0, 0, 1]])

    angles = np.deg2rad(angles)
    rotx = np.array([[1, 0, 0, 0],
                 [0, np.cos(angles[0]), -np.sin(angles[0]), 0],
                 [0, np.sin(angles[0]), np.cos(angles[0]), 0],
                 [0, 0, 0, 1]])
    roty = np.array([[np.cos(angles[1]), 0, np.sin(angles[1]), 0],
                 [0, 1, 0, 0],
                 [-np.sin(angles[1]), 0, np.cos(angles[1]), 0],
                 [0, 0, 0, 1]])
    rotz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, 0],
                 [np.sin(angles[2]), np.cos(angles[2]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
    rotation_matrix = offset_opp.dot(rotz).dot(roty).dot(rotx).dot(offset)
    affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)
    return ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')

def load_data():
    data_dir = '/export/home/elbrion/unet_uda/bladder_160_160_128/base/data'
    ct_train_image = np.zeros((30, *image_size, 1))
    ct_train_mask = np.zeros((30, *image_size, 2))
    ct_test_image = np.zeros((30, *image_size, 1))
    ct_test_mask = np.zeros((30, *image_size, 2))
    cbct_train_image = np.zeros((30, *image_size, 1))
    cbct_train_mask = np.zeros((30, *image_size, 2))
    cbct_test_image = np.zeros((30, *image_size, 1))
    cbct_test_mask = np.zeros((30, *image_size, 2))
    for patient_num in range(30):
        # cts
        ct_train_image[patient_num, :, :, :, 0] = np.load(data_dir + '/ct_train_' + str(patient_num) + '-image.npy')
        ct_train_mask[patient_num, :, :, :, 0] = np.load(data_dir + '/ct_train_' + str(patient_num) + '-mask.npy')
        ct_train_mask[patient_num, :, :, :, 1] = 1 - ct_train_mask[patient_num, :, :, :, 0]
        ct_test_image[patient_num, :, :, :, 0] = np.load(data_dir + '/ct_test_' + str(patient_num) + '-image.npy')
        ct_test_mask[patient_num, :, :, :, 0] = np.load(data_dir + '/ct_test_' + str(patient_num) + '-mask.npy')
        ct_test_mask[patient_num, :, :, :, 1] = 1 - ct_test_mask[patient_num, :, :, :, 0]
        # cbcts
        cbct_train_image[patient_num, :, :, :, 0] = np.load(data_dir + '/cbct_train_' + str(patient_num) + '-image.npy')
        cbct_train_mask[patient_num, :, :, :, 0] = np.load(data_dir + '/cbct_train_' + str(patient_num) + '-mask.npy')
        cbct_train_mask[patient_num, :, :, :, 1] = 1 - cbct_train_mask[patient_num, :, :, :, 0]
        cbct_test_image[patient_num, :, :, :, 0] = np.load(data_dir + '/cbct_test_' + str(patient_num) + '-image.npy')
        cbct_test_mask[patient_num, :, :, :, 0] = np.load(data_dir + '/cbct_test_' + str(patient_num) + '-mask.npy')
        cbct_test_mask[patient_num, :, :, :, 1] = 1 - cbct_test_mask[patient_num, :, :, :, 0]
    return ct_train_image, ct_train_mask, ct_test_image, ct_test_mask, cbct_train_image, cbct_train_mask, cbct_test_image, cbct_test_mask

def load_transformed(e):
    src_dir = '/export/home/elbrion/unet_uda/bladder_160_160_128/base/data_augmented'
    ct_train_image_transformed = np.load(src_dir + '/ct_train_image_transformed_' + str(e) + '.npy')
    ct_train_mask_transformed = np.load(src_dir + '/ct_train_mask_transformed_' + str(e) + '.npy')
    cbct_train_image_transformed = np.load(src_dir + '/cbct_train_image_transformed_' + str(e) + '.npy')

    return ct_train_image_transformed, ct_train_mask_transformed, cbct_train_image_transformed
