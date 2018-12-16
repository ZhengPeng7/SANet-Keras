from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Conv2DTranspose, ReLU, UpSampling2D
from keras.models import Model
from keras_contrib.layers import InstanceNormalization


def conv_module(x, num_channel_out, kernel_sizes=(2, 2), padding='same', strides=(1, 1), transpose=False, IN=False):
    if transpose:
        x = Conv2DTranspose(num_channel_out, kernel_sizes, strides=strides, padding=padding, use_bias=(not IN))(x)
        # Substitute conv_transpose with (upsampling + conv)
        # x = UpSampling2D(size=strides, interpolation='nearest')(x)
        # x = Conv2D(num_channel_out, kernel_sizes, padding=padding, use_bias=(not IN))(x)
    else:
        x = Conv2D(num_channel_out, kernel_sizes, strides=strides, padding=padding, use_bias=(not IN))(x)
    if IN:
        x = InstanceNormalization()(x)
    x = ReLU()(x)
    return x


def conv_parallel(x, num_channel_out, reduction_layer=True, IN=False):
    branches = []
    kernel_sizes = [1, 3, 5, 7]
    num_kernel_sizes = len(kernel_sizes)
    for idx in range(num_kernel_sizes):
        if reduction_layer and idx > 0:
            branch = conv_module(x, num_channel_out*2, (1, 1), IN=IN)
            branch = conv_module(branch, num_channel_out, (kernel_sizes[idx], kernel_sizes[idx]), IN=IN)
        else:
            branch = conv_module(x, num_channel_out, (kernel_sizes[idx], kernel_sizes[idx]), IN=IN)
        branches.append(branch)
    x = concatenate(branches)
    return x


def encoder(x, num_channel_out_lst=[16, 32, 32, 16], IN=False):
    for idx_num_channel_out in range(len(num_channel_out_lst)):
        x = conv_parallel(x, num_channel_out_lst[idx_num_channel_out], reduction_layer=(idx_num_channel_out != 0), IN=IN)
        if idx_num_channel_out < len(num_channel_out_lst) - 1:
            x = MaxPooling2D()(x)
    return x


def decoder(x, IN=False):
    kernel_sizes = [9, 7, 5]
    num_channel_out_lst = [64, 32, 16]
    for idx in range(len(kernel_sizes)):
        x = conv_module(x, num_channel_out_lst[idx], (kernel_sizes[idx], kernel_sizes[idx]), IN=IN)
        x = conv_module(x, num_channel_out_lst[idx], (2, 2), strides=(2, 2), transpose=True, IN=IN)
    x = conv_module(x, 16, (3, 3), IN=IN)
    x = conv_module(x, 1, (1, 1))
    return x


def SANet(input_shape=(None, None, 3)):
    input_flow = Input(input_shape)
    x = encoder(input_flow, IN=False)
    x = decoder(x, IN=False)
    
    model = Model(inputs=input_flow, outputs=x)
    return model
