#######################################
## Author: Aitor González (@AAitorG) ##
#######################################

from math import log2
from tensorflow.keras import Model, layers
from .modules import *

def UNETR_2D(
            input_shape,
            patch_size,
            num_patches,
            projection_dim,
            transformer_layers,
            num_heads,
            transformer_units,
            data_augmentation = None,
            num_filters = 16, 
            num_classes = 1,
            decoder_activation = 'relu',
            decoder_kernel_init = 'he_normal',
            ViT_hidd_mult = 3,
            batch_norm = True,
            dropout = 0.0
        ):

    """
    UNETR architecture adapted for 2D operations. It combines a ViT with U-Net, replaces the convolutional encoder with the ViT
    and adapt each skip connection signal to their layer's spatial dimensionality. 

    Note: Unlike the original UNETR, the sigmoid activation function is used in the last convolutional layer.

    The ViT implementation is based on keras implementation:
        https://keras.io/examples/vision/image_classification_with_vision_transformer/
    Only code:
        https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    UNETR paper:
        https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf
    
    Args:
      input_shape: the shape of the input image.
      patch_size: the size of the patches that are extracted from the input image. As an example, to use 16x16 patches, set patch_size = 16.
                  As each layer doubles the spatial size, this value must be 2^x.
      num_patches: number of patches to extract from the image. Take into account that each patch must be of specified patch_size.
      projection_dim: the dimension of the embedding space.
      transformer_layers: number of transformer encoder layers
      num_heads: number of heads in the multi-head attention layer.
      transformer_units: number of units in the MLP blocks.
      data_augmentation: a function that takes an input tensor and returns an augmented tensor. 
                         To make use of tensorflow additional data augmentation layers 
                         (use tf layer, if multiple layers, then use sequential() and add them, 
                         and use the resulting sequential layer)
      num_filters: number of filters in the first UNETR's layer of the decoder. In each layer the previous number of filters is doubled. Defaults to 16.
      num_classes: number of classes to predict. Is the number of channels in the output tensor. Defaults to 1.
      decoder_activation: activation function for the decoder. Defaults to relu.
      decoder_kernel_init: Initializer for the kernel weights matrix of the convolutional layers in the
                           decoder. Defaults to he_normal
      ViT_hidd_mult: the multiple of the transformer encoder layers from of which the skip connection signal is going to be extracted.
                     As an example, if we have 12 transformer encoder layers, and we set ViT_hidd_mult = 3, we are going to take
                     [1*ViT_hidd_mult, 2*ViT_hidd_mult, 3*ViT_hidd_mult] -> [Z3, Z6, Z9] encoder's signals. Defaults to 3.
      batch_norm: whether to use batch normalization or not. Defaults to True.
      dropout: dropout rate for the decoder (can be a list of dropout rates for each layer).
    
    Returns:
      A UNETR_2D Keras model.
    """
    
    # ViT

    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs) if data_augmentation != None else inputs
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Hidden states
    hidden_states_out = []

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        # save hidden state
        hidden_states_out.append(encoded_patches)

    # UNETR Part (bottom_up, from the bottle-neck, to the output)

    total_upscale_factor = int(log2(patch_size))
    # make a list of dropout values if needed
    if type( dropout ) is float: 
        dropout = [dropout,]*total_upscale_factor

    # bottleneck
    z = layers.Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])(encoded_patches) # use the encoder output (easier to try different things)
    x = up_green_block(z, num_filters * (2**(total_upscale_factor-1)) )

    for layer in reversed(range(1, total_upscale_factor)):
        # skips (with blue blocks)
        z = layers.Reshape([ input_shape[0]//patch_size, input_shape[1]//patch_size, projection_dim ])( hidden_states_out[ (ViT_hidd_mult * layer) - 1 ] )
        for _ in range(total_upscale_factor - layer):
            z = mid_blue_block(z, num_filters * (2**layer), activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        # decoder
        x = layers.concatenate([x, z])
        x = two_yellow(x, num_filters * (2**(layer)), activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        x = up_green_block(x, num_filters * (2**(layer-1)))

    # first skip connection (out of transformer)
    first_skip = two_yellow(augmented, num_filters, activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0]) 
    x = layers.concatenate([first_skip, x])

    # UNETR_2D output 
    x = two_yellow(x, num_filters, activation=decoder_activation, kernel_initializer=decoder_kernel_init, batch_norm=batch_norm, dropout=dropout[0] )
    output = layers.Conv2D( num_classes, (1, 1), activation='sigmoid', name="mask") (x) # semantic segmentation -- ORIGINAL: softmax

    # Create the Keras model.
    model = Model(inputs=inputs, outputs=output)
    return model
