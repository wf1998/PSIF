import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import K

'''
def classifier(name=None,transition_layer_num=0):
    if name == 'vgg19':
        return keras.applications.vgg19.VGG19(include_top=True, weights='imagenet')
    model=keras.models.load_model(r'model/cifar10_classifier_0.7553.h5',
                            custom_objects={'leaky_relu': tf.nn.leaky_relu})
    print(len(model.layers))
    for i,layer in enumerate(model.layers):
        if i<=transition_layer_num:
            layer.trainable=True
        else:
            layer.trainable=False
    return model
'''

encoder_input_shape = (32, 32, 3)
classifier_input_shape = (32, 32, 3)
decoder_flatten = 32 * 32 * 3


def sampling(z_mean, z_log_var):
    epsilon = K.random_normal(shape=(K.shape(z_mean)))
    return z_mean + K.exp(z_log_var * 0.5) * epsilon


class IBModel(keras.Model):
    def __init__(self,latent_dimension = 100,information_ratio=0,transition_layer_num=2):
        super(IBModel, self).__init__()
        # self._name=str((information_ratio+1)*transition_layer_num)
        self.transition_layer_num=transition_layer_num
        self.information_ratio=information_ratio
        self.latent_dimension=latent_dimension

        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=encoder_input_shape, padding="same",
                                         activation=tf.nn.leaky_relu,name='my_conv1')
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv2')
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv3')
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv4')
        self.flat = keras.layers.Flatten(name='flat1')
        self.dense1 = keras.layers.Dense(1000, activation=tf.nn.leaky_relu,name='my_dense1')
        self.dense2 = keras.layers.Dense(500, activation=tf.nn.leaky_relu,name='my_dense2')
        self.dense3 = keras.layers.Dense(self.latent_dimension,name='my_dense3')
        self.dense4 = keras.layers.Dense(self.latent_dimension,name='my_dense4')

        self.dense11 = keras.layers.Dense(self.latent_dimension, activation=tf.nn.leaky_relu,name='my_dense11')
        self.dense12 = keras.layers.Dense(500, activation=tf.nn.leaky_relu,name='my_dense12')
        self.dense13 = keras.layers.Dense(1000, activation=tf.nn.leaky_relu,name='my_dense13')
        self.dense14 = keras.layers.Dense(decoder_flatten, activation=tf.nn.leaky_relu,name='my_dense14')

        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape,name='reshape1')
        self.conv11 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv11')
        self.conv12 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv12')
        self.conv13 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv13')
        self.conv14 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv14')
        self.conv15 = keras.layers.Conv2D(3, kernel_size=3, padding="same", activation=tf.nn.tanh,name='my_conv15')
        self.cloud_model = classifier(transition_layer_num=self.transition_layer_num)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x=self.conv4(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        z_mean = self.dense3(x)
        z_log_var = self.dense4(x)
        samp = sampling(z_mean, z_log_var)
        x=self.dense11(samp)
        x=self.dense12(x)
        x=self.dense13(x)
        x = self.dense14(x)
        x = self.reshape(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv14(x)
        feature = self.conv15(x)*0.5+0.5
        y = self.cloud_model(feature)
        return y

    def model(self):
        x = keras.layers.Input(shape=encoder_input_shape)
        return keras.Model(inputs=x, outputs=self.call(x))
