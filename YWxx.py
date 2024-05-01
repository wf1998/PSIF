import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K


from 源代码.experiment_cifar10 import classifier, encoder_input_shape, sampling


class BIGVAE1(keras.Model):
    def __init__(self):
        super(BIGVAE1, self).__init__()
        self.conv1 = keras.layers.Conv2D(32, kernel_size=3, input_shape=encoder_input_shape, padding="same",
                                         activation=tf.nn.leaky_relu,name='my_conv1')
        self.conv2 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv2')
        self.conv3 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv3')
        self.conv4 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv4')
        self.conv5 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv4_')

        self.reshape = keras.layers.Reshape(target_shape=encoder_input_shape)
        self.conv11 = keras.layers.Conv2D(256, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv11')
        self.conv12 = keras.layers.Conv2D(128, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv12')
        self.conv13 = keras.layers.Conv2D(64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv13')
        self.conv14 = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu,name='my_conv14')
        self.conv15 = keras.layers.Conv2D(3, kernel_size=3, padding="same", activation=tf.nn.tanh,name='my_conv15')
        self.cloud_model = classifier(j=6)
        # self.cloud_model.trainable = True

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x1_=x
        x = self.conv3(x)
        # x2_=x
        z_mean=self.conv4(x)
        z_log_var=self.conv5(x)

        epsilon = sampling(z_mean, z_log_var)
        samp = z_mean + tf.exp(z_log_var / 2) * epsilon
        x = self.conv11(samp)
        # x = 0.7*self.conv12(x)+0.3*x2_
        x = self.conv12(x)
        x = 0.9*self.conv13(x)+0.1*x1_
        # x = (self.conv13(x)+x1_)/2
        x = self.conv14(x)
        feature = self.conv15(x)*0.5+0.5
        # feature=0.99*feature+0.01*inputs
        y = self.cloud_model(feature)
        return y
