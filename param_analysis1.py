import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import experiment_mnist
import os,gc

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
opt = keras.optimizers.Adam(lr=0.0002)
SCC = keras.losses.SparseCategoricalCrossentropy()

def D_and_I(dimention, i, test_x, start_index):
    for index,d in enumerate(dimention):
        model_path='model/new/cifar10_SP_' + d + '_' + i + '_2.h5'
        if os.path.exists(model_path):
            SP_model = keras.models.load_model(model_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
            inputs = SP_model.inputs
            y = SP_model(inputs)
            feature = SP_model.get_layer('my_conv15').output
            SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
            feature=SP_model(test_x[start_index+index].reshape(-1,32,32,3))

            feature_img= (1 - float(i)) * (feature[1].numpy().reshape(32, 32, 3) * 0.5 + 0.5) + float(i) * test_x[start_index+index]
            plt.imshow(test_x[start_index+index])
            plt.show()
            plt.imshow(feature_img)
            plt.show()
            print(i+'_'+d)
            del SP_model
            keras.backend.clear_session()



def D_and_L(dimention, l, test_x, start_index):
    for index,l in enumerate(layer):#cifar10_SP_10_0.01_1
        model_path='model/new/cifar10_SP_' + dimention + '_'+'0.02_' + l + '.h5'
        if os.path.exists(model_path):
            SP_model = keras.models.load_model(model_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
            inputs = SP_model.inputs
            y = SP_model(inputs)
            feature = SP_model.get_layer('my_conv15').output
            SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
            feature=SP_model(test_x[start_index+index].reshape(-1,32,32,3))

            feature_img= 0.98* (feature[1].numpy().reshape(32, 32, 3) * 0.5 + 0.5) + 0.02 * test_x[start_index + index]
            plt.imshow(test_x[start_index+index])
            plt.show()
            plt.imshow(feature_img)
            plt.show()
            print(l + '_' + dimention)
            del SP_model
            keras.backend.clear_session()


def L_and_I(i, layer, test_x, start_index):
    for index,l in enumerate(layer):#cifar10_SP_10_0.01_1
        model_path='model/new/cifar10_SP_100_'+i+'_' + l + '.h5'
        if os.path.exists(model_path):
            SP_model = keras.models.load_model(model_path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
            inputs = SP_model.inputs
            y = SP_model(inputs)
            feature = SP_model.get_layer('my_conv15').output
            SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
            feature=SP_model(test_x[start_index+index].reshape(-1,32,32,3))

            feature_img=  (1 - float(i)) * (feature[1].numpy().reshape(32, 32, 3) * 0.5 + 0.5) + float(i) * test_x[start_index+index]
            plt.imshow(test_x[start_index+index])
            plt.show()
            plt.imshow(feature_img)
            plt.show()
            print(l + '_' + i)
            del SP_model
            keras.backend.clear_session()

if __name__ == '__main__':
    layer = ['1', '2', '3', '4', '5']
    dimention = ['500', '300', '100', '50', '10']
    information = ['0.01', '0.02', '0.03', '0.04', '0.06', '0.08', '0.1']
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
    test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)

    #D_and_I(dimention,information[6],test_x,30)
    # D_and_L(dimention[4], layer, test_x, 50)
    L_and_I(information[6], layer, test_x, 70)

