import tensorflow as tf
import keras.layers as L
from keras.models import Model



def conv_block(c, filters, strides=1):
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.Conv2D(filters, (6, 6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(c)
    c = L.MaxPooling2D((2, 2), strides=(2, 2))(c)
    return c

def deconv_block(d, filters, strides=1):
    d = L.Conv2DTranspose(filters, (6,6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(d)
    d = L.Conv2DTranspose(filters, (6,6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(d)
    d = L.Conv2DTranspose(filters, (6,6), activation='relu', padding='same', strides=strides, kernel_initializer='he_normal')(d)
    d = L.UpSampling2D((2,2))(d)
    return d

def vggEncoder(input_shape):
    inputs = L.Input(shape=input_shape)
    x = conv_block(inputs, filters=16, strides=1)
    x = conv_block(x, filters=32, strides=1)
    x = conv_block(x, filters=64, strides=1)
    x = conv_block(x, filters=128, strides=1)
    x = conv_block(x, filters=256, strides=1)
    x = conv_block(x, filters=512, strides=1)

    encoder_model = Model(inputs=inputs, outputs=x)
    return encoder_model


def vggDecoder(input_shape):
    inputs = L.Input(shape=(8,8,512))
    x = deconv_block(inputs, filters=512, strides=1)
    x = deconv_block(x, filters=256, strides=1)
    x = deconv_block(x, filters=128, strides=1)
    x = deconv_block(x, filters=64, strides=1)
    x = deconv_block(x, filters=32, strides=1)
    x = deconv_block(x, filters=16, strides=1)

    outputs = L.Conv2DTranspose(3, (6,6), activation='sigmoid', padding='same', strides=(1,1), kernel_initializer='he_normal')(x)
    
    decoder_model = Model(inputs=inputs, outputs=outputs)
    return decoder_model