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



