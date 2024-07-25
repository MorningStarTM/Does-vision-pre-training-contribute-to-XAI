import tensorflow as tf
import keras.layers as L
from keras.models import Model



def conv_block(c, filters, strides=1):
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.MaxPooling2D((2, 2), strides=(2, 2))(c)
    return c

def vggEncoder(input_shape):
    first_filter = 16
    inputs = L.Input(shape=input_shape)

    x = L.Conv2D(first_filter, kernel_size=7, strides=1, padding='same')(inputs)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha=0.2)(x)

    for value in range(first_filter, 256 + 1):
        if (value & (value - 1)) == 0 and value >= first_filter:
            x = conv_block(x, filters=value, strides=1)

    encoder_model = Model(inputs, x, name='encoder')
    return encoder_model



def vgg_decoder(inputs:tuple):
    decoder_in = L.Input(shape=(8,8,256))

    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(decoder_in)
    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d = L.Conv2DTranspose(256, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d = L.UpSampling2D((2,2))(d)

    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d)
    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d1 = L.Conv2DTranspose(128, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d1 = L.UpSampling2D((2,2))(d1)

    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d1)
    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d2 = L.Conv2DTranspose(64, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d2 = L.UpSampling2D((2,2))(d2)

    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d2)
    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d3 = L.Conv2DTranspose(32, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d3 = L.UpSampling2D((2,2))(d3)

    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d3)
    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)
    d4 = L.Conv2DTranspose(16, (6,6), activation='relu', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)
    d4 = L.UpSampling2D((2,2))(d4)

    outputs = L.Conv2DTranspose(3, (6,6), activation='sigmoid', padding='same', strides=(1,1), kernel_initializer='he_normal')(d4)

    decoder_model = Model(decoder_in, outputs, name='decoder')
    return decoder_model