import tensorflow as tf
from keras import layers as L
from keras.models import Model
from .resnet import resEncoder, resDecoder
from .vgg import vggEncoder, vggDecoder


class ResAutoencoder:
    def __init__(self, inputs: tuple):
        self.encoder = resEncoder(inputs)
        self.decoder = resDecoder(self.encoder.output)

    def build_autoencoder(self):
        autoencoder_inputs = self.encoder.input
        encoded = self.encoder(autoencoder_inputs)
        autoencoder_outputs = self.decoder(encoded)

        autoencoder_model = Model(autoencoder_inputs, autoencoder_outputs, name='autoencoder')
        return autoencoder_model


class VggAutoencoder:
    def __init__(self, inputs: tuple):
        self.encoder = vggEncoder(inputs)
        self.decoder = vggDecoder(self.encoder.output)

    def build_autoencoder(self):
        autoencoder_inputs = self.encoder.input
        encoded = self.encoder(autoencoder_inputs)
        autoencoder_outputs = self.decoder(encoded)

        autoencoder_model = Model(autoencoder_inputs, autoencoder_outputs, name='autoencoder')
        return autoencoder_model
    
