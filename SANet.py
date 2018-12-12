from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Conv2DTranspose, ZeroPadding2D
from keras.models import Model
from keras_contrib.layers import InstanceNormalization


def conv_parallel(x, num_channel_out, IN=True):
    branches = []
    kernel_sizes = [1, 3, 5, 7]
    num_kernel_sizes = len(kernel_sizes)
    for idx in range(num_kernel_sizes):
        if idx != 1:
            branch = Conv2D(num_channel_out//2, (1, 1), padding='same', activation='relu')(x)
        branch = Conv2D(num_channel_out, (kernel_sizes[idx], kernel_sizes[idx]), padding='same', activation='relu', use_bias=(not IN))(x)
        # branch = ZeroPadding2D(padding=(idx, idx))(branch)
        if IN:
            branches.append(InstanceNormalization()(branch))
        else:
            branches.append(branch)
    x = concatenate(branches)
    return x


def encoder(x, num_channel_out_lst=[16, 32, 32, 16]):
    for num_channel_out in num_channel_out_lst:
        x = conv_parallel(x, num_channel_out)
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    return x


def decoder(x, IN=True):
    kernel_sizes = [9, 7, 5]
    num_channel_out_lst = [64, 32, 16]
    num_kernel_sizes = len(kernel_sizes)
    for idx in range(num_kernel_sizes):
        x = Conv2D(num_channel_out_lst[idx], (kernel_sizes[idx], kernel_sizes[idx]), padding='same', activation='relu', use_bias=(not IN))(x)
        # x = ZeroPadding2D(padding=(1+num_kernel_sizes-idx, 1+num_kernel_sizes-idx))(x)
        if IN:
            x = InstanceNormalization()(x)
        x = Conv2DTranspose(num_channel_out_lst[idx], (2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same', activation='relu', use_bias=(not IN))(x)
    # x = ZeroPadding2D(padding=(1, 1))(x)
    if IN:
        x = InstanceNormalization()(x)
    x = Conv2D(1, (1, 1), padding='same')(x)
    return x


def SANet(input_shape=(None, None, 3)):
    input_flow = Input(input_shape)
    x = encoder(input_flow)
    x = decoder(x)
    
    model = Model(inputs=input_flow, outputs=x)
    return model
