import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


def classifier(name=None,transition_layer_num=0):
    if name=='vgg19':
        return keras.applications.vgg19.VGG19(include_top=True,weights='imagenet')
    model=keras.models.load_model(r'model/fashion_mnist_classifier.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    for i, layer in enumerate(model.layers):
        if i <= transition_layer_num:
            layer.trainable = True
        else:
            layer.trainable = False
    return model


encoder_input_shape = (28, 28, 1)
classifier_input_shape = (28, 28, 1)
decoder_flatten = 28 * 28 * 1
latent_dimension = 100



def sampling(z_mean, z_log_var):
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var * 0.5) * epsilon


class BIGVAE(keras.Model):
    def __init__(self):
        super(BIGVAE, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=encoder_input_shape, padding="same",
                                         activation=tf.nn.leaky_relu,name='my_conv1')
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv2')
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv3')
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv4')
        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500, activation=tf.nn.leaky_relu,name='dense1')
        self.dense2 = keras.layers.Dense(200, activation=tf.nn.leaky_relu,name='dense2')
        self.dense3 = keras.layers.Dense(latent_dimension,name='dense3')
        self.dense4 = keras.layers.Dense(latent_dimension,name='dense4')

        self.dense11 = keras.layers.Dense(latent_dimension, activation=tf.nn.leaky_relu,name='dense11')
        self.dense12 = keras.layers.Dense(200, activation=tf.nn.leaky_relu,name='dense12')
        self.dense13 = keras.layers.Dense(500, activation=tf.nn.leaky_relu,name='dense13')
        self.dense14 = keras.layers.Dense(decoder_flatten, activation=tf.nn.leaky_relu,name='dense14')
        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape)
        self.conv11 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv11')
        self.conv12 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv12')
        self.conv13 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv13')
        self.conv14 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv14')
        self.conv15 = keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=tf.nn.tanh,name='my_conv15')
        self.cloud_model = classifier(transition_layer_num=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(self.conv4(x))
        x = self.dense1(x)
        x = self.dense2(x)
        z_mean = self.dense3(x)
        z_log_var = self.dense4(x)
        epsilon = sampling(z_mean, z_log_var)
        samp = z_mean + tf.exp(z_log_var / 2) * epsilon
        # samp=x
        x = self.dense14(self.dense13(self.dense12(self.dense11(samp))))
        x = self.reshape(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        feature = self.conv15(x)*0.5+0.5
        # feature=0.99*feature+0.01*inputs
        y = self.cloud_model(feature)
        return y

    def model(self):
        x = keras.layers.Input(shape=encoder_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))


class MY(keras.Model):
    def __init__(self):
        super(MY, self).__init__()
        # self._name = 'vae'
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), padding="same",
                                         activation=tf.nn.leaky_relu,name='my_conv1')
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv2')

        self.flat = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(500, activation=tf.nn.leaky_relu, name='dense1')
        self.dense3 = keras.layers.Dense(100, name='dense3')
        self.dense4 = keras.layers.Dense(100, name='dense4')

        self.dense11 = keras.layers.Dense(100, activation=tf.nn.leaky_relu, name='dense11')
        self.dense13 = keras.layers.Dense(500, activation=tf.nn.leaky_relu, name='dense13')
        self.dense14 = keras.layers.Dense(decoder_flatten, activation=tf.nn.leaky_relu, name='dense14')
        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape)
        self.conv13 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv13')
        self.conv14 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv14')
        self.conv15 = keras.layers.Conv2D(1, kernel_size=3, padding="same", activation=tf.nn.tanh,name='my_conv15')
        self.cloud_model = classifier(transition_layer_num=2)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flat(x)
        x = self.dense1(x)
        # z_mean = self.dense3(x)
        # z_log_var = self.dense4(x)
        # epsilon = sampling(z_mean, z_log_var)
        # samp = z_mean + tf.exp(z_log_var / 2) * epsilon
        samp=x
        x = self.dense14(self.dense13(self.dense11(samp)))
        x = self.reshape(x)
        x = self.conv13(x)
        x = self.conv14(x)
        feature = self.conv15(x)*0.5+0.5
        y = self.cloud_model(feature)
        return y

    def model(self):
        x = keras.layers.Input(shape=(28, 28, 1))
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