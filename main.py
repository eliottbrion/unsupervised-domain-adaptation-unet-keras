from uda import *
from utils import *
from uda_save_10 import *
import tensorflow as tf

for l_max in [0.0]:
    den = 2
    params = {'model': 'unet_L9',
              'lr': 1e-4,
              'batch_size': 2,
              'epochs': 100,
              'n_feat_maps': 16,
              'optimizer': 'Adam',
              'loss': 'dice_loss',
              'd_arch': [64//den, 128//den, 256//den, 512//den, 512//den],
              'e1': 0,
              'l_max': l_max,
              'loss': 'dice_loss',
              'psi': 1.,
              'seed': 2,
               }
    dest_dir = 'results/uda'
    #previous_dir = '/export/home/elbrion/unet_uda/bladder_160_160_128/base/results/uda/model_unet_L9_lr_0.0001_batch_size_2_epochs_40_n_feat_maps_16_optimizer_Adam_loss_dice_loss_d_arch_[32, 64, 128, 256, 256]_e1_0_l_max_0.0_psi_1.0_seed_2'
    previous_dir = None

    train_save(dest_dir, previous_dir, params)
    plot_results(dest_dir, params)
    tf.keras.backend.clear_session()

