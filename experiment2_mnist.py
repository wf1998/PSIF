import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import experiment_mnist
import glob

for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu,True)

def showimg(imgs,ys,original_img=None):
    index=0
    for img,y in zip(imgs,ys):
        if original_img is not None:
            plt.imshow(original_img[index].reshape(28,28))
            plt.show()
            index+=1
        plt.imshow(img)
        plt.title(y)
        plt.show()


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
    batch_size = 100
    epoch = 100
    iteration = 600
    model = experiment_mnist.MY().model()
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
                # tv_loss = - total_variation_loss(feature, 2, 1)
                loss = scc_loss  #+ 100*mse_loss + 100*tv_loss

                grads = g.gradient(loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
                if j == 0 or (j + 1) % 5 == 0:
                    y, feature = model(test_x[100:120].reshape(-1, 28, 28, 1))
                    acc_temp = get_acc(y, test_y[100:120])
                    print('%d:loss=%.6f acc=%.4f' % (j + 1, loss.numpy(), acc_temp))
                # y_pre = np.array(np.zeros(shape=(1, 10)))
                # for k in range(200):
                #     y, feature = model(test_x[k * 50:k * 50 + 50])
                #     y_pre = np.append(y_pre, y, axis=0)
                # y_pre = np.delete(y_pre, 0, axis=0)
                # acc_ = get_acc(y_pre, test_y)
                # if acc < acc_ and acc_ > 0.5:
                #     print('模型准确度从%.4f上升到%.4f' % (acc, acc_), end=',')
                #     acc = acc_
                #     model.save('model/AE3_%.4f.h5' % (acc),overwrite=True)
                #     print('模型保存成功。')
                #     if acc_==1:
                #         print('以获得最优模型，结束程序')
                #         break
                # print('\r %d-%d done!' % (i, j + 1), end='')

                # model.evaluate(x=test_x[:1000], y=test_y[:1000])
    index = np.arange(50)
    y, feature = model(test_x[index].reshape(-1, 28, 28, 1))
    showimg(feature.numpy().reshape(-1, 28, 28))
    compare(y, test_y[index])
    print(get_acc(y, test_y[index]))
    # 执行tensorflow内部训练方案
    # model.fit(x=train_x, y=train_y, epochs=100, validation_data=(test_x, test_y))


# 在测试集上进一步一一验证已保存的模型
def evaluate_model(test_x, test_y):
    path_list = glob.glob('model\AE*.h5')
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

def test_optimal(test_x, test_y,model_name='AE_0.9921.h5'):
    indexs=np.arange(start=50,stop=70)
    model=keras.models.load_model('model/'+model_name,custom_objects={'leaky_relu':tf.nn.leaky_relu})
    y_,feature=model(test_x[indexs])
    showimg(feature.numpy().reshape(-1, 28, 28), test_y[indexs],test_x[indexs])
    compare(y_, test_y[indexs])
    print(get_acc(y_, test_y[indexs]))

class MyCallBack(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        c = str(acc) + ',' + str(loss) + '\n'
        with open('record/mnist_traing_SP_new.csv', 'a') as f:
            f.write(c)
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        c =  str(val_acc) + ',' + str(val_loss) + '\n'
        with open('record/mnist_test_SP_new.csv', 'a') as f:
            f.write(c)
# def train_model_by_feature(original_x, original_y):
#     batch_size=500
#
#     model = keras.models.load_model('model/mnist_combined_0.9886.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
#     y_, feature = model(test_x[20:50])
#     for i in range()
#     showimg(feature.numpy().reshape(-1, 28, 28), test_y[20:50])

if __name__ == '__main__':
    #加载mnist数据集
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = (train_x / 255.0).reshape(-1, 28, 28, 1)
    test_x = (test_x / 255.0).reshape(-1, 28, 28, 1)
    # #训练分类器
    # cloud_model=experiment_mnist.Classifier().model()
    # opt = keras.optimizers.Adagrad(lr=0.0002)
    # SCC = keras.losses.SparseCategoricalCrossentropy()
    # cloud_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/mnist_classifier_no_additional_information.h5',
    #                                                  save_best_only=True,
    #                                                  mode='max',
    #                                                  monitor='val_accuracy',
    #                                                  save_weights_only=False,
    #                                                  verbose=1)
    #
    # cloud_model.fit(x=train_x, y=train_y, epochs=300,callbacks=cp_callback,validation_data=(test_x,test_y))
    # 训练组合器
    cloud_model = experiment_mnist.MY().model()
    opt = keras.optimizers.Adagrad(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    cloud_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    cloud_model.summary()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='model/mnist_combined_SP.h5',
                                                     save_best_only=True,
                                                     mode='max',
                                                     monitor='val_accuracy',
                                                     save_weights_only=False,
                                                     verbose=1)

    cloud_model.fit(x=train_x, y=train_y, callbacks=[cp_callback, MyCallBack()], epochs=300,
                    validation_data=(test_x, test_y))
    #训练网络
    # train_combined_model(train_x, train_y, test_x, test_y)
    # evaluate_model(test_x, test_y)
    # test_optimal(test_x, test_y,model_name='AE3_0.9822.h5')


