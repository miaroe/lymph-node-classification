import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPooling3D, Activation, LSTM, Conv1D, Input
from tensorflow.keras.regularizers import l2

# modified from https://github.com/androst/mlmia/blob/master/mlmia/architectures/timingnet.py

def TimingNet(input_shape=None, num_stations=None):
    """ Instantiates the TimingNet model

    Reference:
    - [Detection of Cardiac Events in Echocardiography Using 3D Convolutional Recurrent Neural Networks]
    (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8580137)(IUS 2018)
    """

    img_input = Input(shape=input_shape)
    conv3d_1 = conv3d_block(inputs=img_input, filters=16, kernel_size=(3, 7, 7))
    conv3d_2 = conv3d_block(inputs=conv3d_1, filters=32, kernel_size=(3, 3, 3))
    conv3d_3 = conv3d_block(inputs=conv3d_2, filters=64, kernel_size=(3, 3, 3))
    conv3d_4 = conv3d_block(inputs=conv3d_3, filters=128, kernel_size=(3, 3, 3))
    conv3d_5 = conv3d_block(inputs=conv3d_4, filters=256, kernel_size=(3, 3, 3))

    time_distributed = layers.TimeDistributed(layers.Flatten())(conv3d_5)

    lstm_1 = LSTM(32, return_sequences=True, go_backwards=False, kernel_regularizer=l2(1e-4))(time_distributed)
    lstm_2 = LSTM(32, return_sequences=False, go_backwards=False, kernel_regularizer=l2(1e-4))(lstm_1)

    model = tf.keras.Model(inputs=img_input, outputs=lstm_2, name="timing_net")
    return model

def conv3d_block(inputs, filters, kernel_size, name="conv3d_block"):
    """ Instantiates a Conv3D block

    # Arguments
        inputs: input tensor
        filters: number of filters in the convolutional layer
        kernel_size: size of the convolutional kernel
        name: name of the block

    # Returns
        x: output tensor
    """

    x = Conv3D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.0001), kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling3D(pool_size=(1, 2, 2))(x) # Downsamples the input along its spatial dimensions (depth, height, and width)
    return x
