#######################################
## Author: Aitor González (@AAitorG) ##
#######################################

from math import log2
from tensorflow.keras import Model, layers
from .modules import *

def YNETR_2D(
            input_shape,
            patch_size,
            num_patches,
            projection_dim,
            transformer_layers,
            num_heads,
            transformer_units,
            data_augmentation = None,
            num_filters = 16, # 16 x num_channels
            num_classes = 1,
            activation = 'relu',
            kernel_init = 'he_normal',
            ViT_hidd_mult=3,
            patch_dropout = 0.0,
            batch_norm = False,
            dropout = 0.0
        ):
    
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs) if data_augmentation != None else inputs
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # dropout patches (experimental)
    encoded_patches = tf.keras.layers.Dropout(patch_dropout, noise_shape=(6, 256, 1), seed=42)(encoded_patches) if patch_dropout > 0.0 else encoded_patches
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

    ###########################   UNET encoder
    total_upscale_factor = int(log2(patch_size))
    # make a list of dropout values if needed
    if type( dropout ) is float: 
        dropout = [dropout,]* (total_upscale_factor+1) # +1 (unet bottleneck)
    
    unet_skips = []
    x = augmented
    # ENCODER (top-down)
    for layer in range(total_upscale_factor):
        x = layers.Conv2D(num_filters * (2**(layer)), (3, 3), activation=None, kernel_initializer=kernel_init, padding='same') (x)
        x = layers.BatchNormalization() (x) if batch_norm else x
        x = layers.Activation(activation) (x)
        if dropout[layer] > 0.0:
            x = layers.Dropout(dropout[layer]) (x)
        x = layers.Conv2D(num_filters * (2**(layer)), (3, 3), activation=None, kernel_initializer=kernel_init, padding='same') (x)
        x = layers.BatchNormalization() (x) if batch_norm else x
        x = layers.Activation(activation) (x)

        unet_skips.append(x)

        x = layers.MaxPooling2D((2, 2))(x)

    # U-Net BOTTLENECK
    x = layers.Conv2D(num_filters * (2**(total_upscale_factor)), (3, 3), activation=None, kernel_initializer=kernel_init, padding='same')(x)
    x = layers.BatchNormalization() (x) if batch_norm else x
    x = layers.Activation(activation) (x)
    if dropout[total_upscale_factor] > 0.0:
            x = layers.Dropout(dropout[layer]) (x)
    x = layers.Conv2D(num_filters * (2**(total_upscale_factor)), (3, 3), activation=None, kernel_initializer=kernel_init, padding='same') (x)
    x = layers.BatchNormalization() (x) if batch_norm else x
    unet_bn = layers.Activation(activation) (x)

    ### UNETR convolutional Part (bottom_up)

    # bottleneck
    b = layers.Reshape([   input_shape[0]//patch_size,
                    input_shape[1]//patch_size,
                    projection_dim])(encoded_patches) # use the encoder output (easier to try different things)
    x = layers.concatenate([b, unet_bn], axis = 3, name='BN_concat')
    if dropout[total_upscale_factor] > 0.0:
        x = layers.SpatialDropout2D(dropout[total_upscale_factor], name='BN_Dropout2d')(x) 
    x = up_green_block(x, num_filters * (2**(total_upscale_factor-1)), name='BN_TConv' )

    for layer in reversed(range(1, total_upscale_factor)):
        # skips (with blue blocks)
        z = layers.Reshape([   input_shape[0]//patch_size,
                        input_shape[1]//patch_size,
                        projection_dim])( 
                            hidden_states_out[ (ViT_hidd_mult * layer) - 1 ] 
                            )
        for _ in range(total_upscale_factor - layer):
            z = mid_blue_block(z, num_filters * (2**layer), activation = activation, kernel_initializer = kernel_init, batch_norm=batch_norm, dropout=dropout[layer])
        # decoder
        x = layers.concatenate([z, x, unet_skips[layer]], axis=3)
        if dropout[layer] > 0.0:
            x = layers.SpatialDropout2D(dropout[layer])(x)
        x = two_yellow(x, num_filters * (2**(layer)), activation = activation, kernel_initializer = kernel_init, batch_norm=batch_norm, dropout=dropout[layer] )
        x = up_green_block(x, num_filters * (2**(layer-1)))

    # Long skip connection
    long_skip = two_yellow(augmented, num_filters, activation = activation, kernel_initializer = kernel_init, batch_norm=batch_norm, dropout=dropout[0])
    x = layers.concatenate([long_skip, x, unet_skips[0]], axis=3)
    if dropout[0] > 0.0:
        x = layers.SpatialDropout2D(dropout[0])(x)

    # Unet output 
    x = two_yellow(x, num_filters, activation = activation, kernel_initializer = kernel_init, batch_norm=batch_norm, dropout=dropout[0] )
    output = layers.Conv2D( num_classes, (1, 1), activation='sigmoid', name="mask") (x) # semantic segmentation -- ORIGINAL: softmax

    # Create the Keras model.
    model = Model(inputs=inputs, outputs=output)
    return model