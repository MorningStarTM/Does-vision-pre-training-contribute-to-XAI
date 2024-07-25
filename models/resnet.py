import tensorflow as tf
from keras.models import Model
from keras import layers




def resnet_block(x, filters, strides=1):
    identity = x

    x = layers.Conv2D(filters, kernel_size=5, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(filters, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if strides > 1:
        identity = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')(identity)
        identity = layers.BatchNormalization()(identity)

    x = layers.Add()([x, identity])
    #x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def resnet_deconv_block(x, filters, strides=1):
    identity = x

    x = layers.Conv2DTranspose(filters, kernel_size=5, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2DTranspose(filters, kernel_size=5, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if strides > 1:
        identity = layers.Conv2DTranspose(filters, kernel_size=1, strides=strides, padding='same')(identity)
        identity = layers.BatchNormalization()(identity)

    x = layers.Add()([x, identity])
    #x = layers.Activation('relu')(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x



def resEncoder(input_shape):
    first_filter = 16
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(first_filter, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    for value in range(first_filter, input_shape[0] + 1):
        if (value & (value - 1)) == 0 and value >= first_filter:
            x = resnet_block(x, filters=value, strides=2)

    encoder_model = Model(inputs, x, name='encoder')
    encoder_model.summary()



def resDecoder(input_shape):
    first_filter = 16
    start = 16
    end = 512
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2DTranspose(512, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(512, kernel_size=7, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    for value in range(end, start - 1, -1):
        if (value & (value - 1)) == 0 and value <= end:
            x = resnet_deconv_block(x, filters=value, strides=2)

    x = layers.Conv2D(8, kernel_size=7, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(3, kernel_size=7, strides=1, padding='same', activation='linear')(x)
    decoder_model = Model(inputs=inputs, outputs=x)
    decoder_model.summary()
    