import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import experiment_cifar10
import glob
from keras.utils import layer_utils

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def showimg(imgs,y_pre, y_true, original_img=None,normalization=False):
    index = 0
    if normalization:
        imgs=imgs.numpy()*0.5+0.5
    for img, y in zip(imgs, y_true):
        if original_img is not None:
            plt.imshow(original_img[index].reshape(32, 32,3))
            plt.show()
        plt.imshow(img)
        title='true-pre:'+str(y)+'-'+str(np.where(y_pre[index]==np.max(y_pre[index]))[0][0])
        plt.title(title)
        plt.show()
        index += 1


def get_acc(y_pre, y_true):
    corr = 0
    for index, y_ in enumerate(y_pre):
        if np.where(y_ == np.max(y_))[0][0] == y_true[index]:
            corr += 1
    acc = corr / len(y_true)
    return acc


def compare(y_pre, y_true):
    for index, y_ in enumerate(y_pre):
        print(np.where(y_ == np.max(y_))[0][0], y_true[index], y_.numpy())


# 计算x方向及y方向相邻像素差值，如果有高频花纹，这个值肯定会高，
def high_pass_x_y(image, gap_x, gap_y):
    x_var = image[:, :, gap_x:, :] - image[:, :, :-gap_x, :]
    y_var = image[:, gap_y:, :, :] - image[:, :-gap_y, :, :]

    return x_var, y_var


# 计算总体变分损失
def total_variation_loss(image, gap_x, gap_y):
    x_deltas, y_deltas = high_pass_x_y(image, gap_x, gap_y)
    return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)


# 训练vae与classifier的组合模型
def train_combined_model(train_x, train_y, test_x, test_y):
    batch_size = 1000
    epoch = 10000
    iteration = 50
    model = experiment_cifar10.BIGVAE().model()
    model.summary()
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    MSE = keras.losses.MeanSquaredError()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    acc = 0
    # 执行gradientTape()方案
    for i in range(epoch):
        print('epoch=' + str(i))
        for j in range(iteration):
            with tf.GradientTape() as g:
                start = j * batch_size
                end = start + batch_size
                img_x = train_x[start:end]
                img_y = train_y[start:end]
                y, feature = model(img_x)
                # noises = tf.random.truncated_normal(mean=0, stddev=0.5, shape=feature.numpy().shape)
                # feature = 0.4 * feature + 0.6 * noises
                scc_loss = SCC(img_y, y)
                # mse_loss = - MSE(img_x, feature)
                # tv_loss = - total_variation_loss(feature, 2, 2)
                loss = scc_loss #+ np.exp(0.001*tv_loss)+ np.exp(mse_loss)

                grads = g.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                # if j == 0 or (j + 1) % 5 == 0:
                #     y, feature = model(test_x[100:120].reshape(-1, 28, 28, 1))
                #     acc_temp = get_acc(y, test_y[100:120])
                #     print('%d:loss=%.6f acc=%.4f' % (j + 1, loss.numpy(), acc_temp))
                y_pre = np.array(np.zeros(shape=(1, 10)))
                for k in range(5):
                    y, feature= model(test_x[k * 2000:k * 2000 + 2000])
                    y_pre = np.append(y_pre, y, axis=0)
                y_pre = np.delete(y_pre, 0, axis=0)
                acc_ = get_acc(y_pre, test_y)
                with open('record/cifar10.txt','a') as f:
                    f.write(str(i*50+j)+','+str(acc_)+','+str(loss.numpy())+'\n')
                    f.flush()
                if acc < acc_ and acc_>0.70:
                    index=np.random.randint(low=0,high=100)
                    Y_, fea = model(test_x[index:index+1])
                    showimg(fea, Y_, test_y[index:index+1], test_x[index:index+1])
                    print('模型准确度从%.4f上升到%.4f' % (acc, acc_), end=',')
                    acc = acc_
                    model.save('model/cifar10_combined_%.4f.h5' % (acc), overwrite=True)
                    print('模型保存成功。')
                    if acc_ == 1:
                        print('以获得最优模型，结束程序')
                        break
                print('\r %d-%d done! acc_=%.4f' % (i, j + 1,acc_), end='')

                # model.evaluate(x=test_x[:1000], y=test_y[:1000])
    index = np.arange(50)
    y, feature = model(test_x[index].reshape(-1, 32, 32, 3))
    showimg(feature.numpy().reshape(-1, 32, 32, 3),y,test_x[index])
    compare(y, test_y[index])
    print(get_acc(y, test_y[index]))
    # 执行tensorflow内部训练方案
    # model.fit(x=train_x, y=train_y, epochs=100, validation_data=(test_x, test_y))


# 在测试集上进一步一一验证已保存的模型
def evaluate_model(test_x, test_y):
    path_list = glob.glob('model\引入过渡层\cifar10*.h5')
    acc, optimal_model = 0, None
    for path in path_list:
        model = keras.models.load_model(path, custom_objects={'leaky_relu': tf.nn.leaky_relu})
        y_pre = np.array(np.zeros(shape=(1, 10)))
        for i in range(2):
            y, feature = model(test_x[i * 5000:i * 5000 + 5000])
            y_pre = np.append(y_pre, y, axis=0)
        y_pre = np.delete(y_pre, 0, axis=0)
        acc_ = get_acc(y_pre, test_y)
        print(path.split('\\')[1], acc_)
        if acc < acc_:
            acc = acc_
            optimal_model = path
    print('最优模型为' + optimal_model)

def evalute(test_x,test_y,model_name):
    model = keras.models.load_model('model/' + model_name, custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adagrad(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x,test_y)

def test_optimal(test_x, test_y, model_name):
    indexs = np.arange(start=1, stop=10)
    model = keras.models.load_model('model/' + model_name, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    y_, feature = model(test_x[indexs])
    showimg(feature.numpy().reshape(-1, 32, 32,3),y_, test_y[indexs], test_x[indexs])
    compare(y_, test_y[indexs])
    print(get_acc(y_, test_y[indexs]))

class MyCallBack(keras.callbacks.Callback):
    def __init__(self,latent_dimension,information_ratio=0,transition_layer_num=2):
        super(MyCallBack, self).__init__()
        self.latent_dimension=latent_dimension
        self.information_ratio=information_ratio
        self.information_ratio=transition_layer_num
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        # acc = logs.get('accuracy')
        # loss = logs.get('loss')
        # c1 = str(acc) + ',' + str(loss) + '\n'
        # with open('record/引入过渡层/new/cifar10_vgg19_traing_SP.csv', 'a') as f:
        #     f.write(c1)
        c =  str(val_acc) + ',' + str(val_loss) + '\n'
        with open('record/引入过渡层/new/cifar10_test_'+str(self.latent_dimension)+'_'+str(information_ratio)+'_'+str(transition_layer_num)+'.csv', 'a') as f:
            f.write(c)

# def train_model_by_feature(original_x, original_y):
#     batch_size=500
#
#     model = keras.models.load_model('model/mnist_combined_0.9886.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
#     y_, feature = model(test_x[20:50])
#     for i in range()
#     showimg(feature.numpy().reshape(-1, 28, 28), test_y[20:50])

if __name__ == '__main__':
    # 加载mnist数据集
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
    test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)
    # 训练分类器
    information_ratio = 0
    transition_layer_num = 2
    cloud_model = experiment_cifar10.BIGVAE(latent_dimension=300,information_ratio=information_ratio,transition_layer_num=transition_layer_num).model()

    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    cloud_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='DP_model/DP_cifar10_{val_accuracy:.4f}.h5',
                                                     save_best_only=True,
                                                     mode='max',
                                                     monitor='val_accuracy',
                                                     save_weights_only=False,
                                                     verbose=1)

    cloud_model.fit(x=train_x, y=train_y, epochs=1000, callbacks=cp_callback, validation_data=(test_x, test_y))

    #训练分类器
    # for information_ratio in [0.061]:
    # # for information_ratio in [0.02,0.03,0.04,0.06,0.08,0.1]:
    # #     for transition_layer_num in range(2,3):
    # #         for latent_dimension in [500,300,100,50,10]:
    #     for transition_layer_num in range(1,6):
    #         for latent_dimension in [100]:
    #             print('latent_dimension=%d,informa  tion_ratio=%.2f,transition_layer_num=%d'%(latent_dimension,information_ratio,transition_layer_num))
    #             opt = keras.optimizers.Adam(learning_rate=0.0001)
    #             SCC = keras.losses.SparseCategoricalCrossentropy()
    #             cloud_model = experiment_cifar10.BIGVAE(latent_dimension,information_ratio,transition_layer_num).model()
    #             cloud_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    #             trainable_count = layer_utils.count_params(cloud_model.trainable_weights)-267491359
    #             cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/new/cifar10_SP_'+str(latent_dimension)+'_'+str(information_ratio)+'_'+str(transition_layer_num)+'.h5',
    #                                                              save_best_only=True,
    #                                                              mode='max',
    #                                                              monitor='val_accuracy',
    #                                                              save_weights_only=False,
    #                                                              verbose=1)
    #             es_callback=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10)
    #             cloud_model.fit(x=train_x, y=train_y,callbacks=[cp_callback,es_callback,MyCallBack(latent_dimension,information_ratio,transition_layer_num)], epochs=200, validation_data=(test_x, test_y))
    #             cloud_model.fit(x=train_x, y=train_y,batch_size=500,callbacks=[cp_callback], epochs=1, validation_data=(test_x, test_y))

    # information_ratio = 0.02
    # transition_layer_num = 2
    # cloud_model = experiment_cifar10.BIGVAE(latent_dimension=300,information_ratio=information_ratio,transition_layer_num=transition_layer_num).model()
    # opt = keras.optimizers.Adam(lr=0.0001)
    # SCC = keras.losses.SparseCategoricalCrossentropy()
    # cloud_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='paramter analysis/model/vgg19_test.h5',
    #                                                  save_best_only=True,
    #                                                  mode='max',
    #                                                  monitor='val_accuracy',
    #                                                  save_weights_only=False,
    #                                                  verbose=1)
    #
    # cloud_model.fit(x=train_x, y=train_y, callbacks=[cp_callback], epochs=300,
    #                 validation_data=(test_x, test_y))
    # 训练网络
    #train_combined_model(train_x, train_y, test_x, test_y)
    # evaluate_model(test_x, test_y)
    # evalute(train_x, train_y,model_name='cifar10_combined_no_additional_information.h5')
    # test_optimal(test_x, test_y,model_name='cifar10_combined_land.h5')
    # (1000-500-300 67.7%) （1000，500，300,0.1orign,0.7489）（1000，500，400,0.15orign,0.7561）
    # （1000，500,0.15orign,0.7549）（1000，500，50,0.15orign,0.7491）（1000，500，100,Tconv,0.75）