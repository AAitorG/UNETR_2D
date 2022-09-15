import tensorflow as tf
from tensorflow.keras import layers

# Transformer utilities

class Patches(layers.Layer):
    # It takes a batch of images and returns a batch of patches
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    # It takes patches and projects them into a `projection_dim` dimensional space, then the position embedding is added
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
def mlp(x, hidden_units, dropout_rate):
    """
    It takes an input tensor and returns a tensor that is the result of applying a transformer multi-layer
    perceptron (MLP) block to the input
    
    Args:
      x: The input layer.
      hidden_units: A list of integers, the number of units for each mlp hidden layer. 
                    It defines the dimensionality of the output space at each mlp layer
      dropout_rate: The dropout rate to use.
    
    Returns:
      The output of the last layer.
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

'''
UNETR_2D BLOCKS

To make easier to read, same blocks described in the UNETR architecture are defined below, but using 2D operations.
    UNETR paper:
        https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf
'''

def basic_yellow_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor, applies a convolutional layer with the specified number of
    filters, applies batch normalization, applies an activation function, and applies dropout if
    specified.
    
    Args:
      x: the input tensor
      filters: the number of filters in the convolutional layer
      activation: the activation function to use. Defaults to relu
      kernel_initializer: This is the initializer for the kernel weights matrix (see initializers).
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: the dropout rate
    
    Returns:
      The output of the last layer in the block.
    """
    x = layers.Conv2D(filters, (3,3), padding = 'same', kernel_initializer = kernel_initializer)(x)
    x = layers.BatchNormalization() (x) if batch_norm else x
    x = layers.Activation(activation) (x)
    x = layers.Dropout(dropout)(x) if dropout > 0.0 else x
    return x

def up_green_block(x, filters, name=None):
    """
    This function takes in a tensor and a number of filters and returns a tensor that is the result of
    applying a 2x2 transpose convolution with the given number of filters.
    
    Args:
      x: the input tensor
      filters: The number of filters for the transpose convolutional layer.
      name: The name of the layer (optional).
    
    Returns:
      The output of the Conv2DTranspose layer.
    """
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same', name=name) (x)
    return x

def mid_blue_block(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor and returns an output tensor after applying a transpose convolution which upscale x2 the spatial size, 
    and applies a convolutional layer.
    
    Args:
      x: the input tensor
      filters: number of filters in the convolutional layers
      activation: The activation function to use. Defaults to relu
      kernel_initializer: Initializer for the convolutional kernel weights matrix (see initializers).
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: The dropout rate.
    
    Returns:
      The output of the last layer of the block.
    """
    x = up_green_block(x, filters)
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)
    return x
    
def two_yellow(x, filters, activation='relu', kernel_initializer='glorot_uniform', batch_norm=True, dropout=0.0):
    """
    This function takes in an input tensor, and returns an output tensor that is the result of
    applying two basic yellow blocks to the input tensor.
    
    Args:
      x: the input tensor
      filters: number of filters in the convolutional layer
      activation: The activation function to use. Defaults to relu
      kernel_initializer: Initializer for the kernel weights matrix (see initializers). 
                          Defaults to glorot_uniform
      batch_norm: Whether to use batch normalization or not. Defaults to False
      dropout: The dropout rate.
    
    Returns:
      The output of the second basic_yellow_block.
    """
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=dropout)
    x = basic_yellow_block(x, filters, activation=activation, kernel_initializer=kernel_initializer, batch_norm=batch_norm, dropout=0.0)
    return x