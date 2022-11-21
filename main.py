from utils import *
from models import *

import os
import numpy as np
import cv2
import gc
import pandas as pd
from matplotlib import pyplot as plt
from skimage import metrics
from datetime import datetime
from time import time 

import tensorflow as tf
import tensorflow_addons as tfa # AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from tensorflow.keras.backend import clear_session

set_gpu(gpu_id=0)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

set_seed()

# ## Parameters
data_path = '../Datasets/'
# number of different experiments (EACH LIST VALUE IS ONE EXPERIMENT,)
num_exec = 1
# starting experiment
starts = 0
# number of repetitions for each experiment
repetitions = 2
# name of each execution
_exec_id = [ 'UNETR_2D-base',]* num_exec
# output file name
csv_out_dir = './csv_results/'
csv_filename = csv_out_dir + 'UNETR_2D-base-date-{}-results.csv'.format(str(datetime.now()).replace(':', '_'))
# Training datasets
_train_datasets = [['Lucchi++',],] * num_exec
# Test datasets
_test_datasets = [['Lucchi++',],] * num_exec

## === Training parameters ===
# number of epochs
_numEpochs = [360,] * num_exec 
# patience (if (patience <= 0): patience will not be used)
_patience = [60,] * num_exec
# learning rate
_lr = [1e-4,] * num_exec
# weight_decay  (for AdamW)
_wd = [1e-5,] * num_exec
# Scheduler: 'oneCycle', 'reduce', 'cosine',  None
_schedule = ['cosine',] * num_exec
# Optimizer name: 'Adam', 'SGD', 'rmsprop', 'AdamW'
_optimizer_name = ['AdamW',] * num_exec
# Loss function name: 'bce', 'bce_dice', 'mse'
_loss_acronym = ['bce',] * num_exec
# batch size
_batch_size_value = [ 6, ] * num_exec

## === Network parameters ===
# Network architecture: UNETR_2D, YNETR_2D,
_model_name = [ 'UNETR_2D',] * num_exec
# initial filters (16 x num_channels)
_num_filters = [32,] * num_exec
# conv kernel initializer: 'glorot_uniform', 'he_normal'
_kernel_init = ['he_normal',] * num_exec
# conv part activation function
_activation = ['relu',] * num_exec
# patch size
_patch_size = [ 16,] * num_exec
# hidden dimension
_hidden_dim = [ 256,] * num_exec #256 - base
# number of transformer encoders
_transformer_layers = [ 12, ] * num_exec
# number of heads per MHA module
_num_heads = [ 12, ] * num_exec
# transformer mlp dimentions
_mlp_dim = [ [1024, 256], ] * num_exec
# number of output channels (number of classes)
_out_channels = [1,] * num_exec
# denoise type: cutout, gaussNoise, coarseSaltP, emulate_LR, gaussian_filter, defocusBlur, motionBlur, pixeldropout
_posible_dataAug = [[],] * num_exec
# dropout value # [0.1, 0.1, 0.2, 0.2, 0.3]
_dropout = [ 0.0,] * num_exec
# multiple of ViT layers that will be used for each skip connection
_ViT_hidd_mult_skipC = [ 3, ] * num_exec
# Use Batch Normalization layers
_batch_norm = [True,] * num_exec
# Use Data Augmentation
_da = [True,] * num_exec
# tensorflow additional data augmentation layers (use tf layer, if multiple layers, then use sequential() and add them)
_extra_tf_data_augmentation = [None,] * num_exec

# %%
# === Extra parameters ===
# Load weights for FineTunning
_use_saved_model = [False] * num_exec
# Path to save weights (After training)
_out_dir = ['./model_weights',] * num_exec
# Path to save plots
img_out_dir = './plots'
# filenames for trained model weights (h5)
_weights_filename = []
for i in range(num_exec):
    for r in range(repetitions):
        out_name = 'weights-{}-id-{}-rep-{}-src-{}-bce-nf-{}-bs-{}-{}-{}.h5'.format(
                                                    _model_name[i], _exec_id[i], r, '_'.join(_train_datasets[i]), _num_filters[i], _batch_size_value[i],
                                                    _optimizer_name[i], 'None' if _schedule[i] is None else _schedule[i] )
        _weights_filename.append(os.path.join(_out_dir[i], out_name))
# Weights file path (weights that will be loaded if use_saved_model is TRUE)
_model_path = list(_weights_filename) # * num_exec

# image input size (this does not change the data size!)
_input_shape = [(256,256,1),] * num_exec
# number of random patches (with number lower than 0, sequential patches will be used)
_n_patches = [-1,] * num_exec

# evaluation parameters
# patch size
patch_h, patch_w = (256,256)
# relevant patch size
relevant_h, relevant_w = (128,128)


def main():

    
    all_test_datasets = set()
    for d_r in _test_datasets: all_test_datasets.update(set(d_r))
    posible_datasets = set(all_test_datasets)
    for d_r in _train_datasets: posible_datasets.update(set(d_r))
    
    all_test_datasets = list(all_test_datasets)
    posible_datasets = list(posible_datasets)
    
    all_results = get_empty_all_results(posible_datasets)

    for e in range(starts, num_exec):

        set_seed()
        
        iou_per_r = {d:[] for d in all_test_datasets }

        for r in range(repetitions):

            ####################################### 
            ############# PARAMETERS ##############
            #######################################

            ## === Training parameters ===
            numEpochs = _numEpochs[e]
            patience = _patience[e]
            lr = _lr[e]
            wd = _wd[e]
            schedule = _schedule[e]
            optimizer_name = _optimizer_name[e]
            loss_acronym = _loss_acronym[e]
            batch_size_value = _batch_size_value[e]
            ## === Network parameters ===
            model_name = _model_name[e]
            num_filters = _num_filters[e]
            kernel_init = _kernel_init[e]
            activation = _activation[e]
            patch_size = _patch_size[e]
            hidden_dim = _hidden_dim[e]
            transformer_layers = _transformer_layers[e]
            num_heads = _num_heads[e]
            mlp_dim = _mlp_dim[e]
            extra_tf_data_augmentation = _extra_tf_data_augmentation[e]
            out_channels = _out_channels[e]
            # === Extra parameters ===
            use_saved_model = _use_saved_model[e]
            model_path = _model_path.pop(0) if use_saved_model else ''
            out_dir = _out_dir[e]
            input_shape = _input_shape[e]
            n_patches = _n_patches[e]
            weights_filename = _weights_filename.pop(0)
            ViT_hidd_mult_skipC = _ViT_hidd_mult_skipC[e]
            posible_dataAug = _posible_dataAug[e]
            train_datasets = _train_datasets[e]
            test_datasets = _test_datasets[e]
            batch_norm = _batch_norm[e]
            da = _da[e]
            dropout = _dropout[e]

            exec_id = '{}_{}-Rep{}-{}'.format(e+1, num_exec, r, _exec_id[e])

            print( "\n-------{:^20}-------\n".format("TRAIN  "+'{}/{}: {} \t Rep:{}/{}'.format(e+1, num_exec, _exec_id[e], r+1, repetitions)  ))

            ####################################### 
            ############### GET DATA ##############
            ####################################### 

            h_cuts = 0
            v_cuts = 0
            
            input_images = []
            gt_labels = []

            for train_ds in train_datasets:

                source_path = os.path.join(data_path, train_ds, 'train')
                ds_imgs, ds_lbls = get_xy_image_list(source_path)

                assert len(ds_imgs) > 0, 'There in NO data, check path: {}.'.format(source_path)
                assert len(ds_imgs) == len(ds_lbls), 'There is different ammount of images and labels. Images: {}  Labels: {}'.format(len(ds_imgs), len(ds_lbls))

                h, w = ds_imgs[0].shape
                exp_h, exp_w, _ = input_shape
                
                h_cuts = int(np.ceil(w/exp_w))
                v_cuts = int(np.ceil(h/exp_h))

                #print("h_cuts: {} \t v_cuts: {}".format(h_cuts, v_cuts))

                if w%exp_w != 0 and h%exp_h != 0:
                    w_parts = w/exp_w
                    h_parts = h/exp_h
                    new_w = int(np.ceil(w_parts))*exp_w
                    new_h = int(np.ceil(h_parts))*exp_h
                    # MIRROR PADDING
                    ds_imgs = [mirror_border(x, new_h, new_w) for x in ds_imgs]
                    ds_lbls = [mirror_border(x, new_h, new_w) for x in ds_lbls]
                    # ZERO PADDING (for 256x256 patches by default)
                    #ds_imgs = [add_padding(x) for x in ds_imgs]
                    #ds_lbls = [add_padding(x) for x in ds_lbls]

                if n_patches < 0:
                    # sequential patches
                    ds_imgs = create_patches( ds_imgs, h_cuts, v_cuts )
                    ds_lbls = create_patches( ds_lbls, h_cuts, v_cuts )

                    #ds_imgs, ds_lbls = filter_patches(ds_imgs, ds_lbls)
                else:
                    # random patches
                    p_ds_imgs = []
                    p_ds_lbls = []
                    while len(p_ds_imgs)<n_patches:
                        a,b = create_random_patches( ds_imgs, ds_lbls, 1, [256, 256] )
                        #a, b = filter_patches(a, b)
                        p_ds_imgs = p_ds_imgs + a
                        p_ds_lbls = p_ds_lbls + b
                    ds_imgs = p_ds_imgs[:n_patches]
                    ds_lbls = p_ds_lbls[:n_patches]

                ds_imgs = np.expand_dims(ds_imgs, axis=-1)
                ds_lbls = np.expand_dims(ds_lbls, axis=-1)

                if len(input_images) == 0:
                    input_images = np.array(ds_imgs)
                    gt_labels = np.array(ds_lbls)
                else:
                    input_images = np.concatenate([input_images, ds_imgs], axis=0)
                    gt_labels = np.concatenate([gt_labels, ds_lbls], axis=0)

            train_data_size = input_images.shape[0] * 0.9
            val_data_size = input_images.shape[0] * 0.1

            print('\n Data shape:',input_images.shape)

            crappify = get_crappify(posible_dataAug) # get crappify function

            '''
            if e == 0:
                gt_labels = [getSobels(img[:,:,0])*255 for img in input_images]
                gt_labels = np.expand_dims(gt_labels, axis=-1)
            '''

            # Domain 1: Source
            train_generator, val_generator = get_train_val_generators(  X_data = input_images,
                                                                        Y_data = input_images if loss_acronym == 'mse' else gt_labels,
                                                                        validation_split = 0.1,
                                                                        rescale = 1./255,
                                                                        horizontal_flip=True if da else False,
                                                                        vertical_flip=True if da else False,
                                                                        rotation_range = 180 if da else 0,
                                                                        #width_shift_range=0.2,
                                                                        #height_shift_range=0.2,
                                                                        #shear_range=0.2,
                                                                        batch_size=batch_size_value,
                                                                        show_examples=False,
                                                                        preprocessing_function = crappify if loss_acronym == 'mse' else None,
                                                                        val_preprocessing_function = crappify if loss_acronym == 'mse' else None, )

            ####################################### 
            ############### COMPILE ###############
            #######################################

            # Free up RAM in case the model definition cells were run multiple times
            clear_session()
            gc.collect()

            ### CALLBACKS ###
            callbacks = []
            
            if patience > 0:
                # callback for early stop
                earlystopper = EarlyStopping(patience=patience, verbose=1, restore_best_weights=True)
                callbacks.append(earlystopper)

            if schedule == 'oneCycle':
                # callback for one-cycle schedule
                steps = np.ceil(train_data_size / batch_size_value) * numEpochs
                #steps = np.ceil(len(X_train) / batch_size_value) * numEpochs
                lr_schedule = OneCycleScheduler(lr, steps)
                callbacks.append(lr_schedule)
            elif schedule == 'reduce':
                # callback to reduce the learning rate in the plateau
                lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                        patience=patience, min_lr=(lr/10))
                callbacks.append(lr_schedule)
            elif schedule == 'cosine':
                # this scheduler is not a callback
                steps = np.ceil(train_data_size / batch_size_value) * numEpochs
                lr = tf.keras.optimizers.schedules.CosineDecay(lr, steps)  


            # create the network and compile it with its optimizer
            if model_name == 'UNETR_2D':
                model = UNETR_2D(
                        input_shape = input_shape,
                        patch_size = patch_size,
                        num_patches = (input_shape[0]**2)//(patch_size**2),
                        projection_dim = hidden_dim,
                        transformer_layers = transformer_layers,
                        num_heads = num_heads,
                        transformer_units = mlp_dim, 
                        data_augmentation = extra_tf_data_augmentation,
                        num_filters = num_filters,
                        num_classes = out_channels,
                        decoder_activation = activation,
                        decoder_kernel_init = kernel_init,
                        ViT_hidd_mult=ViT_hidd_mult_skipC,
                        batch_norm = batch_norm,
                        dropout = dropout,
                    )
            elif model_name == 'YNETR_2D':   
                model = YNETR_2D(
                            input_shape = input_shape,
                            patch_size = patch_size,
                            num_patches = (input_shape[0]**2)//(patch_size**2),
                            projection_dim = hidden_dim,
                            transformer_layers = transformer_layers,
                            num_heads = num_heads,
                            transformer_units = mlp_dim, 
                            data_augmentation = extra_tf_data_augmentation,
                            num_filters = num_filters, 
                            num_classes = out_channels,
                            activation = activation,
                            kernel_init = kernel_init,
                            ViT_hidd_mult=ViT_hidd_mult_skipC,
                            batch_norm = batch_norm,
                            dropout = dropout,
                        )
                    

            if optimizer_name == 'SGD':
                optim =  tf.keras.optimizers.SGD(
                        lr=lr, momentum=0.99, decay=0.0, nesterov=False)
            elif optimizer_name == 'Adam':
                optim = tf.keras.optimizers.Adam( learning_rate=lr )
            elif optimizer_name == 'rmsprop':
                optim = tf.keras.optimizers.RMSprop( learning_rate=lr )
            elif optimizer_name == 'AdamW':
                optim = tfa.optimizers.AdamW( weight_decay = wd, learning_rate=lr )
            
            #model.summary()

            if loss_acronym == 'bce':
                loss_funct = 'binary_crossentropy'
            elif loss_acronym == 'bce_dice':
                loss_funct = bce_dice_loss
            elif loss_acronym == 'mse': # dont change this acronym (is used to know when is training for denoising)
                loss_funct = 'mean_squared_error'

            if loss_acronym == 'mse':
                eval_metric = [psnr, ssim]
            else:
                eval_metric = jaccard_index     
                            
            # compile the model with the specific optimizer, loss function and metric
            model.compile(optimizer=optim, loss=loss_funct, metrics=eval_metric)

            if use_saved_model:
                # Restore the weights
                model.load_weights(model_path) # change this

                # compile the model with the specific optimizer, loss function and metric
                model.compile(optimizer=optim, loss=loss_funct, metrics=eval_metric)
                print("Weights loaded, and compiled")


            ### N-Parameters ###
            trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
            non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

            ####################################### 
            ############### TRAIN #################
            #######################################

            train_start_time = time()
            history = model.fit(train_generator, validation_data=val_generator,
                                validation_steps=np.ceil(val_data_size/batch_size_value),
                                steps_per_epoch=np.ceil(train_data_size/batch_size_value),
                                epochs=numEpochs, callbacks=callbacks)
            train_time = (time() - train_start_time)

            ####################################### 
            ############ PLOT AND SAVE ############
            #######################################
            track_metrics = ['loss', 'psnr', 'ssim'] if loss_acronym == 'mse' else ['loss', 'jaccard_index']
            curve_name = exec_id + '_' + '&'.join(test_datasets) + '_loss-metric_curve.png'
            create_dir(img_out_dir)
            plot_loss_and_metric(track_metrics, history, figsize=(14,7), save_fig_path=os.path.join(img_out_dir, curve_name))

            # Save weights for future reuse
            create_dir(out_dir)
            model.save_weights( weights_filename )
            print( 'Saved model as ' + weights_filename )

            ####################################### 
            ############ EVALUATION ###############
            #######################################

            ### FULL IMAGE using relevant patch + sliding window + mirror padding
            for ds in test_datasets:
                print('TEST: ', ds)
                source_test_data_path = os.path.join(data_path, ds, 'test')
                test_img, test_lbl = get_xy_image_list(source_test_data_path)

                # Prepare the test data
                X_test = [x/255 for x in test_img] # normalize between 0 and 1
                X_test = np.expand_dims( np.asarray(X_test, dtype=np.float32), axis=-1 ) # add extra dimension
                Y_test = [x/255 for x in test_lbl] # normalize between 0 and 1
                Y_test = np.expand_dims( np.asarray(Y_test, dtype=np.float32), axis=-1 ) # add extra dimension
                
                b, h, w, c = X_test.shape
                #print("test shape = b:{}, h:{}, w:{}, c:{}".format(b, h, w, c))
                del test_img, test_lbl

                # Now, we calculate the final test metrics
                test_iou = []
                test_psnr = []
                test_ssim = []
                test_mse = []
                inference_time = []
                preds_test = []
                input_test = []

                for i,img in enumerate(X_test):

                    h, w, _ = img.shape

                    if w%relevant_w != 0 and h%relevant_h != 0:
                        w_parts = w/relevant_w
                        h_parts = h/relevant_h
                        new_w = int(np.ceil(w_parts))*relevant_w
                        new_h = int(np.ceil(h_parts))*relevant_h
                        pad_h = new_h - h # if pad==11 (odd): 6 top (near 0) - 5 bot
                        pad_w = new_w - w # if pad==11 (odd): 6 L   (near 0) - 5 R
                        same_shape_windows = False
                    else:
                        new_h = h + relevant_h
                        new_w = w + relevant_w
                        pad_h = relevant_h
                        pad_w = relevant_w
                        same_shape_windows = True

                    image = np.expand_dims(mirror_border(img[:,:,0], new_h, new_w), axis=-1)

                    rows = []
                    x_rows = []
                    # crete patches of (patch_h, patch_w) with (relevant_h, relevant_w) overlap between them
                    for j in range(0, image.shape[0]-relevant_h, relevant_h): 

                        is_first_column = j == 0
                        is_last_column = j == (image.shape[0]-relevant_h*2)

                        columns = [] # patches of the first row
                        for k in range(0, image.shape[1]-relevant_w, relevant_w):
                            window = image[j:j + patch_h, k:k + patch_w, :]
                            columns.append( window )
                        columns = np.array(columns)
                        
                        # prepare input and gt (y)
                        if loss_acronym == 'mse':
                            y = img[:,:,0]
                            columns = np.array(columns*255, dtype='uint8')
                            columns = crappify(columns)
                            columns = np.array(columns, dtype='float32')/255
                        else:
                            y = Y_test[i,:,:,0]

                        inference_time_start = time()
                        _preds_test = model.predict_on_batch(columns)#, batch_size=columns.shape[0])
                        inference_time.append((time()-inference_time_start)/columns.shape[0])

                        if same_shape_windows:
                            #all the patches contain the same padding so we can extract them directly
                            relevant_windows = _preds_test[ :, relevant_h//2 : patch_h-(relevant_h//2),
                                                            relevant_w//2 : patch_w-(relevant_w//2), :]

                            x_relevant = columns[ :, relevant_h//2 : patch_h-(relevant_h//2),
                                                    relevant_w//2 : patch_w-(relevant_w//2), :]
                        else:
                            # if pad==11 (odd): 5 top (near 0) - 6 bot
                            # if pad==11 (odd): 6 L   (near 0) - 5 R

                            # pad_h//2 padding in the top side  &&  round(pad_h/2)-1 padding in the bottom side
                            from_row = pad_h//2 if is_first_column else relevant_h//2
                            to_row = -round(pad_h/2) if is_last_column else patch_h-(relevant_h//2) 

                            # remove especial (smaller) padding in the top and bottom side, of the first or last row
                            relevant_windows = _preds_test[ :, from_row : to_row, relevant_w//2 : patch_w - (relevant_w//2), :]
                            x_relevant = columns[ :, from_row : to_row, relevant_w//2 : patch_w - (relevant_w//2), :]
                            

                            # convert into list otherwise numpy raise an error due to the shape differences
                            relevant_windows = [im for im in relevant_windows]
                            x_relevant = [im for im in x_relevant]


                            # remove especial (smaller) padding in the sides
                            # the relevant window contain round(pad_w/2) size padding in the left side
                            from_column_L = round(pad_w/2)
                            to_column_L = patch_w-(relevant_w//2)
                            # the relevant window contain (pad_w//2) size padding in the right side
                            from_column_R = relevant_w//2
                            to_column_R = -(pad_w//2)
                            
                            # first column (left padding)
                            relevant_windows[0] = _preds_test[0, from_row : to_row, from_column_L : to_column_L, :]
                            x_relevant[0] = columns[0, from_row : to_row, from_column_L : to_column_L, :]
                            # last column (right padding)
                            relevant_windows[-1] = _preds_test[-1, from_row : to_row, from_column_R : to_column_R, :]
                            x_relevant[-1] = columns[-1, from_row : to_row, from_column_R : to_column_R, :]
                        
                        rows.append(cv2.hconcat(relevant_windows)) # append relevant complete row
                        x_rows.append(cv2.hconcat(x_relevant))

                    x_recons = cv2.vconcat(x_rows)
                    input_test.append(x_recons)
                    recons_parts = cv2.vconcat(rows)
                    preds_test.append(recons_parts) # append complete image

                    if loss_acronym == 'mse':
                        test_psnr.append(metrics.peak_signal_noise_ratio(recons_parts, y))
                        test_ssim.append(metrics.structural_similarity(recons_parts, y))
                        test_mse.append(metrics.mean_squared_error(recons_parts, y))
                    else:
                        test_iou.append( jaccard_index(y, recons_parts >= 0.5 ))           
                    
                if loss_acronym == 'mse':
                    mean_iou = None
                    mean_psnr = np.mean(test_psnr)
                    mean_ssim = np.mean(test_ssim)
                    mean_mse = np.mean(test_mse)
                    print("\nTest PSNR:", mean_psnr)
                    print("\nTest SSIM:", mean_ssim)
                    print("\nTest MSE:", mean_mse)

                    display_list1 = [input_test[0], X_test[0,:,:,0], preds_test[0]]
                    display_list2 = [input_test[-1], X_test[-1,:,:,0], preds_test[-1]]
                    display_titles = ['Input full image', 'Ground Truth', 'Predicted image']
                else:
                    mean_iou = np.mean(test_iou)
                    mean_psnr = None
                    mean_ssim = None
                    mean_mse = None
                    print("\nTest IoU:", mean_iou)

                    iou_per_r[ds].append( mean_iou )

                    display_list1 = [X_test[0,:,:,0], Y_test[0,:,:,0], preds_test[0]>=.5, preds_test[0]]
                    display_list2 = [X_test[-1,:,:,0], Y_test[-1,:,:,0], preds_test[-1]>=.5, preds_test[-1]]
                    display_titles = ['Input Image', 'True Mask', 'Predicted Mask', 'Raw Predicted Mask']

                
                first_pred_img_name = exec_id + '-trg-'+ ds + '_first pred img.png'
                last_pred_img_name = exec_id + '-trg-'+ ds + '_last pred img.png'
                plt.figure(figsize=(7*4,7))
                display(display_list1,
                        custom_size = True,
                        save_fig_path = os.path.join(img_out_dir,first_pred_img_name),
                        title = display_titles,
                        )
                plt.figure(figsize=(7*4,7))
                display(display_list2,
                        custom_size = True,
                        save_fig_path = os.path.join(img_out_dir,last_pred_img_name),
                        title = display_titles,
                        )
                del preds_test

                all_results['IoU_' + ds].append( mean_iou )
                all_results['SSIM_' + ds].append( mean_ssim )
                all_results['PSNR_' + ds].append( mean_psnr )
                all_results['MSE_' + ds].append( mean_mse )
                all_results['IoU_' + ds + '_mean'].append( np.mean(iou_per_r[ds]) )
                all_results['IoU_' + ds + '_std'].append( np.std(iou_per_r[ds]) )

                del X_test, Y_test

            # datasets not used
            for ds in set(posible_datasets) - set(test_datasets):
                all_results['IoU_' + ds].append( '' )
                all_results['SSIM_' + ds].append( '' )
                all_results['PSNR_' + ds].append( '' )
                all_results['MSE_' + ds].append( '' )

            ####################################### 
            ############## TO CSV #################
            #######################################

            ### save exec as csv row
            all_results['exec_id'].append( exec_id )
            all_results['repetition'].append(r)
            all_results['model_name'].append( model_name )
            all_results['numEpochs'].append( numEpochs )
            all_results['patience'].append( patience )
            all_results['train_datasets'].append( train_datasets )
            all_results['test_datasets'].append( test_datasets )
            all_results['lr'].append( lr if type( lr ) is float else _lr[e] )
            all_results['wd'].append( wd )
            all_results['schedule'].append( schedule )
            all_results['optimizer_name'].append( optimizer_name )
            all_results['loss_acronym'].append( loss_acronym )
            all_results['batch_size_value'].append( batch_size_value )
            all_results['num_filters'].append( num_filters )
            all_results['kernel_init'].append( kernel_init )
            all_results['activation'].append( activation )
            all_results['patch_size'].append( patch_size )
            all_results['hidden_dim'].append( hidden_dim )
            all_results['transformer_layers'].append( transformer_layers )
            all_results['num_heads'].append( num_heads )
            all_results['mlp_dim'].append( mlp_dim )
            all_results['extra_tf_data_augmentation'].append( 'None' if extra_tf_data_augmentation == None else extra_tf_data_augmentation )
            all_results['out_channels'].append( out_channels )
            all_results['use_saved_model'].append( use_saved_model )
            all_results['used_model_path'].append( '' if use_saved_model else model_path )
            all_results['out_dir'].append( out_dir )
            all_results['input_shape'].append( input_shape )
            all_results['n_rand_patches'].append( 'sequential' if n_patches<0 else n_patches )
            all_results['image curve_name'].append( curve_name )
            all_results['model checkpoint weights out_name'].append( weights_filename )
            all_results['train_time_sec'].append( train_time )
            all_results['mean_inference_time_sec'].append( np.mean(inference_time) )
            all_results['std_inference_time_sec'].append( np.std(inference_time) )
            all_results['trainable_params'].append( trainable_count)
            all_results['non_trainable_params'].append( non_trainable_count)
            all_results['batch_norm'].append(batch_norm)
            all_results['data_augmentation'].append(da)
            all_results['dropout'].append(dropout)
            all_results['skip_layers_mult'].append(ViT_hidd_mult_skipC)

            create_dir(csv_out_dir)
            df=pd.DataFrame(all_results)
            df.to_csv(csv_filename)
        
        print('results in:', csv_filename)

if __name__ == '__main__':
    main()
    print('\n\n################### THE END ###################\n')