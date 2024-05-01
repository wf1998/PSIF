import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

encoder_input_shape = (28, 28, 1)
classifier_input_shape = (28, 28, 1)
decoder_flatten = 28 * 28 * 1
latent_dimension = 100


class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=encoder_input_shape, padding="same",
                                         activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense2 = keras.layers.Dense(10, activation=tf.nn.leaky_relu)
        self.dense3 = keras.layers.Dense(latent_dimension)
        self.dense4 = keras.layers.Dense(latent_dimension)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(self.conv4(x))
        x = self.dense2(self.dense1(x))
        z_mean = self.dense3(x)
        z_log_var = self.dense4(x)
        return z_mean, z_log_var

    def model(self):
        x = keras.layers.Input(shape=encoder_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))



def sampling(z_mean, z_log_var):
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var*0.5) * epsilon


class BIGVAE(keras.Model):
    def __init__(self):
        super(BIGVAE, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=encoder_input_shape, padding="same",
                                         activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense2 = keras.layers.Dense(10, activation=tf.nn.leaky_relu)
        self.dense3 = keras.layers.Dense(latent_dimension)
        self.dense4 = keras.layers.Dense(latent_dimension)

        self.dense11 = keras.layers.Dense(latent_dimension, activation=tf.nn.leaky_relu)
        self.dense12 = keras.layers.Dense(10, activation=tf.nn.leaky_relu)
        self.dense13 = keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense14 = keras.layers.Dense(decoder_flatten, activation=tf.nn.leaky_relu)
        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape)
        self.conv11 = keras.layers.Conv2DTranspose(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv12 = keras.layers.Conv2DTranspose(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv13 = keras.layers.Conv2DTranspose(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv14 = keras.layers.Conv2DTranspose(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv15 = keras.layers.Conv2DTranspose(1, kernel_size=3, padding="same", activation=tf.nn.tanh)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(self.conv4(x))
        x = self.dense2(self.dense1(x))
        z_mean = self.dense3(x)
        z_log_var = self.dense4(x)
        epsilon = sampling(z_mean, z_log_var)
        samp = z_mean + tf.exp(z_log_var / 2) * epsilon
        x = self.dense14(self.dense13(self.dense12(self.dense11(samp))))
        x = self.reshape(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        y = self.conv15(x)
        return y

    def model(self):
        x = keras.layers.Input(shape=encoder_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))


class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = keras.layers.Dense(latent_dimension, activation=tf.nn.leaky_relu)
        self.dense2 = keras.layers.Dense(10, activation=tf.nn.leaky_relu)
        self.dense3 = keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense4 = keras.layers.Dense(decoder_flatten, activation=tf.nn.leaky_relu)
        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape)
        self.conv1 = keras.layers.Conv2DTranspose(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv2 = keras.layers.Conv2DTranspose(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv3 = keras.layers.Conv2DTranspose(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv4 = keras.layers.Conv2DTranspose(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu)
        self.conv5 = keras.layers.Conv2DTranspose(1, kernel_size=3, padding="same", activation=tf.nn.tanh)

    def call(self, inputs, training=None, mask=None):
        x = self.dense4(self.dense3(self.dense2(self.dense1(inputs))))
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        y = self.conv5(x)
        return y

    def model(self):
        x = keras.layers.Input(shape=latent_dimension)
        return keras.Model(inputs=x, outputs=self.call(x))


class Classifier(keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=classifier_input_shape,
                                         activation=tf.nn.leaky_relu)
        self.batchnorm1 = keras.layers.BatchNormalization()
        self.drop1 = keras.layers.Dropout(0.3)
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.leaky_relu)
        self.batchnorm2 = keras.layers.BatchNormalization()
        self.drop2 = keras.layers.Dropout(0.3)
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, activation=tf.nn.leaky_relu)
        self.batchnorm3 = keras.layers.BatchNormalization()
        self.drop3 = keras.layers.Dropout(0.3)
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, activation=tf.nn.leaky_relu)
        self.batchnorm4 = keras.layers.BatchNormalization()
        self.drop4 = keras.layers.Dropout(0.3)
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500, activation=tf.nn.leaky_relu)
        self.dense2 = keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.drop1(self.batchnorm1(self.conv1(inputs)))
        x = self.drop2(self.batchnorm2(self.conv2(x)))
        x = self.drop3(self.batchnorm3(self.conv3(x)))
        x = self.flat(self.drop4(self.batchnorm4(self.conv4(x))))
        y = self.dense2(self.dense1(x))
        # y = self.dense2(x)
        return y

    def model(self):
        x = keras.layers.Input(shape=classifier_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))


class VAE(keras.Model):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder().model()
        self.decoder = Decoder().model()

    def call(self, inputs, training=None, mask=None):
        z_mean, z_log_var = self.encoder(inputs)
        samp = sampling(z_mean, z_log_var)
        y = self.decoder(samp)
        return y

    def model(self):
        x = keras.layers.Input(shape=encoder_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))
