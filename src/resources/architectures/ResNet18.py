
import einops
import tensorflow as tf
from tensorflow.keras import layers

#https://www.tensorflow.org/tutorials/video/video_classification


def conv2Plus1D(filters, kernel_size, padding):
    """
        A sequence of convolutional layers that first apply the convolution operation over the
        spatial dimensions, and then the temporal dimension.
    """
    return tf.keras.Sequential([
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                        kernel_size=(1, kernel_size[1], kernel_size[2]),
                        padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters,
                        kernel_size=(kernel_size[0], 1, 1),
                        padding=padding)
        ])


def residualMain(filters, kernel_size):
    """
        Residual block of the model with convolution, layer normalization, and the
        activation function, ReLU.
    """
    return tf.keras.Sequential([
        conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

def project(units):
    """
        Project certain dimensions of the tensor as the data is passed through different
        sized filters and downsampled.
      """

    return tf.keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = residualMain(filters,
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = project(out.shape[-1])(res)

  return layers.add([res, out])


def resizeVideo(video, target_height, target_width):
    # Extract the shape components for dynamic reshaping
    batch_size, num_frames = tf.shape(video)[0], tf.shape(video)[1]
    height, width, channels = video.shape[2], video.shape[3], video.shape[4]

    # Flatten the batch and time dimensions
    flat_video = tf.reshape(video, [-1, height, width, channels])

    # Resize the frames
    resized_flat_video = layers.Resizing(target_height, target_width)(flat_video)

    # Restore the original batch and time dimensions
    resized_video = tf.reshape(resized_flat_video,
                               [batch_size, num_frames, target_height, target_width, channels])

    return resized_video


def ResNet18(input_shape=None, num_stations=None):

    # get height and width from input shape (None, height, width, channels)
    height, width = input_shape[1], input_shape[2]


    input = layers.Input(shape=input_shape)
    x = input

    x = conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = resizeVideo(x, height // 2, width // 2)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = resizeVideo(x, height // 4, width // 4)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = resizeVideo(x, height // 8, width // 8)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = resizeVideo(x, height // 16, width // 16)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_stations, activation='softmax')(x)

    model = tf.keras.Model(input, x)
    return model



