import numpy as np
import os
from glob import glob
from skimage import io
from skimage.util import img_as_ubyte
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.filters import gaussian
from skimage.util import random_noise
import imgaug.augmenters as iaa
from skimage.transform import rescale, rotate

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_xy_image_list(dir):
    if dir[-1]=='/':
        dir = dir[:-1]
    # Paths to the training images and their corresponding labels
    train_input_path = dir + '/x/*.*'
    train_label_path = dir + '/y/*.*'

    # Read the list of file names
    train_input_filenames = glob(train_input_path)
    train_input_filenames.sort()

    train_label_filenames = glob(train_label_path)
    train_label_filenames.sort()

    print( 'Input images loaded: {} -- Label images loaded: {}\n\tpath: {}'.format(len(train_input_filenames), len(train_label_filenames), dir) )

    # read training images and labels
    train_img = [ img_as_ubyte( np.array(io.imread( x ), dtype='uint8') ) for x in train_input_filenames ]
    train_lbl = [ img_as_ubyte( np.array(io.imread( x ), dtype='uint8') ) for x in train_label_filenames ]
    
    return train_img, train_lbl

def create_patches( imgs, num_x_patches, num_y_patches ):
    ''' Create a list of images patches out of a list of images
    Args:
        imgs: list of input images
        num_x_patches: number of patches in the X axis
        num_y_patches: number of patches in the Y axis
        
    Returns:
        list of image patches
    '''
    original_size = imgs[0].shape
    patch_width = original_size[ 0 ] // num_y_patches
    patch_height = original_size[ 1 ] // num_x_patches
    
    patches = []
    for n in range( 0, len( imgs ) ):
        image = imgs[ n ]
        for i in range( 0, num_y_patches ):
            for j in range( 0, num_x_patches ):
                patches.append( image[ i * patch_width : (i+1) * patch_width,
                                      j * patch_height : (j+1) * patch_height ] )
    return patches

# We define a method to create an arbitrary number of random crops of
# a given size
def create_random_patches( input_imgs, lbl_imgs, num_patches,
                          shape ):
    ''' Create a list of images patches out of a list of images
    Args:
        input_imgs: list of input images
        lbl_imgs: list of input images
        num_patches (int): number of patches for each image.
        shape (2D array): size of the LR patches. Example: [128, 128].
        
    Returns:
        list of image patches and patches of corresponding labels
    '''

    # read training images
    img = input_imgs[0]

    original_size = img.shape
    
    input_patches = []
    label_patches = []
    for n in range( 0, len( input_imgs ) ):
        img = input_imgs[n]
        lbl = lbl_imgs[n]
        for i in range( num_patches ):
          r = np.random.randint(0,original_size[0]-shape[0])
          c = np.random.randint(0,original_size[1]-shape[1])
          input_patches.append(  img[ r : r + shape[0],
                                  c : c + shape[1] ] )
          label_patches.append( lbl[ r : r + shape[0],
                                  c : c + shape[1] ] )
    return input_patches, label_patches

def emulate_LR(images, scale=4, upsample=True):
    xn = np.array(images)
    xorig_max = xn.max()
    xn = xn.astype(np.float32)
    xn /= float(np.iinfo(np.uint8).max)
    xn = random_noise(xn, mode='gaussian', mean=0.0, var=0.01)
    new_max = xn.max()
    x = xn
    if new_max > 0:
        xn /= new_max
    xn *= xorig_max
    multichannel = len(x.shape) > 2
    x = rescale(x, scale=1/scale, order=1, multichannel=multichannel)
    if upsample:
        x = rescale(x, scale=scale, order=1, multichannel=multichannel)
    return x.astype(np.uint8)

def gaussian_filter(images):
    im = np.asarray(images, dtype=np.float32)
    im = gaussian(im, sigma=3, preserve_range=True)
    im = np.asarray(im, dtype=np.uint8)
    return im

def cutout(images):
    aug1 = iaa.Cutout(nb_iterations=(1, 5), size=0.2, cval=0)

    im = np.asarray(images, dtype=np.float32)/255
    im = aug1(images=im)
    im = np.rint(im * 255)
    return im
    
def get_crappify(posible_dataAug):
    def crappify(imgs):
        def _crappify(img):
            img = img.astype(np.uint8)
            i = np.random.randint(low = 0, high = len(posible_dataAug))
            aug = posible_dataAug[i]
            im = aug(images=img)
            return np.array(im, dtype=np.float32)

        if len(posible_dataAug) == 0:
            resul = imgs
        else:
            if len(imgs.shape) == 4:
                resul = np.array([_crappify(img[:,:,0]) for img in imgs])
                resul = np.expand_dims(resul, axis=-1)
            elif len(imgs.shape) == 3:
                resul = _crappify(imgs[:,:,0])
                resul = np.expand_dims(resul, axis=-1)
            elif len(imgs.shape) == 2:
                resul = _crappify(imgs)
        return resul
    return crappify

gaussNoise = iaa.AdditiveGaussianNoise(scale=0.15*255)
coarseSaltP = iaa.CoarseSaltAndPepper(0.2, size_px=(4,16))
defocusBlur = iaa.imgcorruptlike.DefocusBlur(severity=2)
motionBlur = iaa.MotionBlur(k=15)
pixeldropout = iaa.Dropout(p=(0, 0.2))

# %%

def add_padding(np_img):
    '''
    Given a numpy array, add padding to the image so that the image is a multiple of 256x256
    
    Args:
      np_img: the image to be padded
    
    Returns:
      A numpy array of the image with the padding added.
    '''

    image = Image.fromarray(np_img)
    height, width = np_img.shape

    if not width%256 and not height%256:
        return np_img
    
    x = width/256
    y = height/256

    new_width = int(np.ceil(x))*256
    new_height = int(np.ceil(y))*256

    left = int( (new_width - width)/2 )
    top = int( (new_height - height)/2 )
    
    result = Image.new(image.mode, (new_width, new_height), 0)
    
    result.paste(image, (left, top))

    return np.asarray(result)

def remove_padding(np_img, out_shape):
    '''
    Given an image and the shape of the original image, remove the padding from the image
    
    Args:
      np_img: the image to remove padding from
      out_shape (int,int): the desired shape of the output image (height, width)
    
    Returns:
      The image with the padding removed.

    Note:
        If returned image contain any 0 in the shape may be due to the given shape is greater than actual image shape
    '''

    height, width = out_shape # original dimensions
    pad_height, pad_width = np_img.shape # dimensions with padding

    if not width%256 and not height%256: # no hacia falta padding --> no tiene
        return np_img
    
    rm_left = int( (pad_width - width)/2 )
    rm_top = int( (pad_height - height)/2 )

    rm_right = pad_width - width - rm_left
    rm_bot = pad_height - height - rm_top

    return np.array(np_img[rm_top:-rm_bot, rm_left:-rm_right])

import cv2
def reconstruct_images(images, num_x_patches, num_y_patches):
    """
    input:
        ndarray[] images: sequential patches from as much images as you want.
        int num_x_patches: Number of patches per row.
        int num_y_patches: Number of patches per column.
    
    return: list of reconstructed images. Combination of given patches.
    """

    results = []
    imgs = list(images)
    while len(imgs)>0:
        y_patches = []
        for i in range(num_y_patches):
            x_patches = []
            for j in range(num_x_patches):
                x_patches.append(imgs.pop(0))
            y_patches.append(cv2.hconcat(x_patches))
        results.append(cv2.vconcat(y_patches))
    return results

# %%
def filter_patches(img_list, lbl_list, zeros_perc = 0.5):
    resul_img_list = []
    resul_lbl_list = []
    for img, lbl in zip(img_list, lbl_list):
        img_hist,_ = np.histogram(np.array(img).ravel(), bins=np.arange(256))
        #print(img_hist[0]/np.sum(img_hist))
        if img_hist[0]/np.sum(img_hist) < zeros_perc : # 0s are less than the x% of the image
            # only_filt_zeros
            resul_img_list.append(img)
            resul_lbl_list.append(lbl)
    return resul_img_list, resul_lbl_list
#source_img_f, source_lbl_f = filter_patches(source_img, source_lbl)

# %%
def mirror_border(image, sizeH, sizeW):
    h_res = sizeH - image.shape[0]
    w_res = sizeW - image.shape[1]

    top = bot = h_res // 2
    left = right = w_res // 2
    top += 1 if h_res % 2 != 0 else 0
    left += 1 if w_res % 2 != 0 else 0

    res_image = np.pad(image, ((top, bot), (left, right)), 'symmetric')
    return res_image

# Random rotation of an image by a multiple of 90 degrees
def random_90rotation( img ):
    return rotate(img, 90*np.random.randint( 0, 5 ), preserve_range=True)

# Runtime data augmentation
def get_train_val_generators(X_data, Y_data, validation_split=0.25,
                             batch_size=32, seed=42, rotation_range=0,
                             horizontal_flip=True, vertical_flip=True,
                             width_shift_range=0.0,
                             height_shift_range=0.0,
                             shear_range=0.0,
                             brightness_range=None,
                             rescale=None,
                             preprocessing_function=None,
                             val_preprocessing_function = None,
                             show_examples=False):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1-validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed, shuffle=False)

    if not val_preprocessing_function is None:
        X_test = np.array([val_preprocessing_function(x) for x in X_test])
    
    # Image data generator distortion options
    data_gen_args_X = dict( rotation_range = rotation_range,
                          width_shift_range=width_shift_range,
                          height_shift_range=height_shift_range,
                          shear_range=shear_range,
                          brightness_range=brightness_range,
                          preprocessing_function=preprocessing_function,
                          horizontal_flip=horizontal_flip,
                          vertical_flip=vertical_flip,
                          rescale = rescale,
                          fill_mode='reflect')
    
    # Image data generator distortion options
    data_gen_args_Y = dict( rotation_range = rotation_range,
                          width_shift_range=width_shift_range,
                          height_shift_range=height_shift_range,
                          shear_range=shear_range,
                          brightness_range=brightness_range,
                          preprocessing_function=None,
                          horizontal_flip=horizontal_flip,
                          vertical_flip=vertical_flip,
                          rescale = rescale,
                          fill_mode='reflect')


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args_X)
    Y_datagen = ImageDataGenerator(**data_gen_args_Y)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
     
    
    # Validation data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator(rescale=rescale)
    Y_datagen_val = ImageDataGenerator(rescale=rescale)
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=False, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=False, seed=seed)
    
    if show_examples:
        plt.figure(figsize=(10,10))
        # generate samples and plot
        for i in range(3):
            # define subplot
            plt.subplot(321 + 2*i)
            # generate batch of images
            batch = X_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
            plt.subplot(321 + 2*i+1)
            # generate batch of images
            batch = Y_train_augmented.next()
            # convert to unsigned integers for viewing
            image = batch[0]
            # plot raw pixel data
            plt.imshow(image[:,:,0], vmin=0, vmax=1, cmap='gray')
        # show the figure
        plt.show()
        X_train_augmented.reset()
        Y_train_augmented.reset()
    
    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)
    
    return train_generator, test_generator

def get_empty_all_results(posible_datasets):
    all_results = {}
    all_results['exec_id'] = []
    all_results['repetition'] = []
    for ds in set(posible_datasets):
        all_results['IoU_' + ds] = []
        all_results['IoU_' + ds + '_mean'] = []
        all_results['IoU_' + ds + '_std'] = []
    for ds in set(posible_datasets):
        all_results['SSIM_' + ds] = []
        all_results['PSNR_' + ds] = []
        all_results['MSE_' + ds] = []
    
    #all_results['IoU'] = []
    #all_results['PSNR'] = []
    #all_results['SSIM'] = []
    #all_results['MSE'] = []
    all_results['train_datasets'] = []
    all_results['test_datasets'] = []
    all_results['numEpochs'] = []
    all_results['patience'] = []
    all_results['lr'] = []
    all_results['wd'] = []
    all_results['schedule'] = []
    all_results['optimizer_name'] = []
    all_results['loss_acronym'] = []
    all_results['batch_size_value'] = []
    all_results['model_name'] = []
    all_results['num_filters'] = []
    all_results['patch_size'] = []
    all_results['hidden_dim'] = []
    all_results['transformer_layers'] = []
    all_results['num_heads'] = []
    all_results['mlp_dim'] = []
    all_results['extra_tf_data_augmentation'] = []
    all_results['out_channels'] = []
    all_results['batch_norm'] = []
    all_results['data_augmentation'] = []
    all_results['dropout'] = []
    all_results['skip_layers_mult'] = []
    all_results['input_shape'] = []
    all_results['use_saved_model'] = []
    all_results['n_rand_patches'] = []
    all_results['kernel_init'] = []
    all_results['activation'] = []
    all_results['image curve_name'] = []
    all_results['out_dir'] = []
    all_results['model checkpoint weights out_name'] = []
    all_results['used_model_path'] = []
    all_results['train_time_sec'] = []
    all_results['mean_inference_time_sec'] = []
    all_results['std_inference_time_sec'] = []
    all_results['trainable_params'] = []
    all_results['non_trainable_params'] = []
    return all_results