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
    first_filter = 16
    inputs = L.Input(shape=input_shape)

    x = L.Conv2D(first_filter, kernel_size=7, strides=1, padding='same')(inputs)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha=0.2)(x)

    for value in range(first_filter, input_shape[0] + 1):
        if (value & (value - 1)) == 0 and value >= first_filter:
            x = conv_block(x, filters=value, strides=1)

    encoder_model = Model(inputs, x, name='encoder')
    return encoder_model

def vggDecoder(tensor):
    input_shape = tf.keras.backend.int_shape(tensor)
    start = 16
    end = input_shape[3]
    inputs = L.Input(shape=input_shape[1:])

    x = L.Conv2DTranspose(end, kernel_size=7, strides=1, padding='same')(inputs)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha=0.2)(x)

    for value in range(end, start - 1, -1):
        if (value & (value - 1)) == 0 and value <= end:
            x = deconv_block(x, filters=value, strides=1)

    x = L.Conv2D(8, kernel_size=7, strides=1, padding='same', activation='relu')(x)
    x = L.Conv2D(3, kernel_size=7, strides=1, padding='same', activation='linear')(x)
    
    decoder_model = Model(inputs=inputs, outputs=x)
    return decoder_model