from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dense, Conv2DTranspose, ReLU, UpSampling2D, Activation
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.initializers import RandomNormal


def SANet(IN=True):
    input_flow = Input((None, None, 3))

    # Module 1
    channel_out_1 = 16
    x_1 = Conv2D(channel_out_1, 1, padding='same', use_bias=(not IN))(input_flow)
    x_1 = Activation('relu')(x_1)
    x_1 = InstanceNormalization()(x_1) if IN else x_1

    x_2 = Conv2D(channel_out_1*2, 1, padding='same', use_bias=(not IN))(input_flow)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2
    x_2 = Conv2D(channel_out_1, 3, padding='same', use_bias=(not IN))(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2

    x_3 = Conv2D(channel_out_1*2, 1, padding='same', use_bias=(not IN))(input_flow)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3
    x_3 = Conv2D(channel_out_1, 5, padding='same', use_bias=(not IN))(x_3)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3

    x_4 = Conv2D(channel_out_1*2, 1, padding='same', use_bias=(not IN))(input_flow)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4
    x_4 = Conv2D(channel_out_1, 7, padding='same', use_bias=(not IN))(x_4)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4

    x = concatenate([x_1, x_2, x_3, x_4])

    x = MaxPooling2D()(x)

    # Module 2
    channel_out_2 = 32
    x_1 = Conv2D(channel_out_2, 1, padding='same', use_bias=(not IN))(x)
    x_1 = Activation('relu')(x_1)
    x_1 = InstanceNormalization()(x_1) if IN else x_1

    x_2 = Conv2D(channel_out_2*2, 1, padding='same', use_bias=(not IN))(x)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2
    x_2 = Conv2D(channel_out_2, 3, padding='same', use_bias=(not IN))(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2

    x_3 = Conv2D(channel_out_2*2, 1, padding='same', use_bias=(not IN))(x)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3
    x_3 = Conv2D(channel_out_2, 5, padding='same', use_bias=(not IN))(x_3)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3

    x_4 = Conv2D(channel_out_2*2, 1, padding='same', use_bias=(not IN))(x)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4
    x_4 = Conv2D(channel_out_2, 7, padding='same', use_bias=(not IN))(x_4)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4

    x = concatenate([x_1, x_2, x_3, x_4])

    x = MaxPooling2D()(x)

    # Module 3
    channel_out_3 = 32
    x_1 = Conv2D(channel_out_3, 1, padding='same', use_bias=(not IN))(x)
    x_1 = Activation('relu')(x_1)
    x_1 = InstanceNormalization()(x_1) if IN else x_1

    x_2 = Conv2D(channel_out_3*2, 1, padding='same', use_bias=(not IN))(x)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2
    x_2 = Conv2D(channel_out_3, 3, padding='same', use_bias=(not IN))(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2

    x_3 = Conv2D(channel_out_3*2, 1, padding='same', use_bias=(not IN))(x)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3
    x_3 = Conv2D(channel_out_3, 5, padding='same', use_bias=(not IN))(x_3)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3

    x_4 = Conv2D(channel_out_3*2, 1, padding='same', use_bias=(not IN))(x)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4
    x_4 = Conv2D(channel_out_3, 7, padding='same', use_bias=(not IN))(x_4)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4

    x = concatenate([x_1, x_2, x_3, x_4])

    x = MaxPooling2D()(x)

    # Module 4
    channel_out_4 = 32
    x_1 = Conv2D(channel_out_4, 1, padding='same', use_bias=(not IN))(x)
    x_1 = Activation('relu')(x_1)
    x_1 = InstanceNormalization()(x_1) if IN else x_1

    x_2 = Conv2D(channel_out_4*2, 1, padding='same', use_bias=(not IN))(x)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2
    x_2 = Conv2D(channel_out_4, 3, padding='same', use_bias=(not IN))(x_2)
    x_2 = Activation('relu')(x_2)
    x_2 = InstanceNormalization()(x_2) if IN else x_2

    x_3 = Conv2D(channel_out_4*2, 1, padding='same', use_bias=(not IN))(x)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3
    x_3 = Conv2D(channel_out_4, 5, padding='same', use_bias=(not IN))(x_3)
    x_3 = Activation('relu')(x_3)
    x_3 = InstanceNormalization()(x_3) if IN else x_3

    x_4 = Conv2D(channel_out_4*2, 1, padding='same', use_bias=(not IN))(x)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4
    x_4 = Conv2D(channel_out_4, 7, padding='same', use_bias=(not IN))(x_4)
    x_4 = Activation('relu')(x_4)
    x_4 = InstanceNormalization()(x_4) if IN else x_4

    x = concatenate([x_1, x_2, x_3, x_4])

    # Decoder

    x = Conv2D(64, 9, padding='same', use_bias=(not IN))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    x = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    x = Conv2D(32, 7, padding='same', use_bias=(not IN))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    x = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    x = Conv2D(16, 5, padding='same', use_bias=(not IN))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    x = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x


    # Output

    x = Conv2D(16, 3, padding='same', use_bias=(not IN))(x)
    x = Activation('relu')(x)
    x = InstanceNormalization()(x) if IN else x

    # x = Conv2D(16, 5, padding='same', use_bias=(not IN))(x)
    # x = Activation('relu')(x)
    # x = InstanceNormalization()(x) if IN else x

    x = Conv2D(1, 1)(x)
    x = Activation('relu')(x)

    model = Model(inputs=input_flow, outputs=x)

    return model
