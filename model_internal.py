import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu,True)

def showimg(imgs, ys, original_img=None):
    index = 0
    for img, y in zip(imgs, ys):
        if original_img is not None:
            plt.imshow(original_img[index].reshape(28, 28))
            plt.show()
            index += 1
        plt.imshow(img)
        plt.title(y)
        plt.show()

def get_lable_from_distribution(prediction):
    lables=[]
    for y in prediction:
        lable=np.where(y==np.max(y))[0][0]
        lables.append(lable)
    return lables

def DP_cifar10():
    content='budget,acc\n'
    for bugget in np.linspace(0.01,1,99):
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        # train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
        test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)
        noise = np.random.laplace(loc=0, scale=bugget, size=(10000, 32, 32, 3))

        coin = np.random.random(size=(10000, 32, 32, 3))
        flip = np.where(coin < 1, 1, 0)  # 0.6的概率添加噪声
        noise = noise * flip

        test_x += noise
        test_x = np.clip(test_x, a_min=0, a_max=1)
        model = keras.models.load_model('model/cifar10_classifier_0.7553.h5',
                                        custom_objects={'leaky_relu': tf.nn.leaky_relu})
        opt = keras.optimizers.Adam(lr=0.0001)
        SCC = keras.losses.SparseCategoricalCrossentropy()
        model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
        acc=model.evaluate(test_x, test_y)[1]
        content+=str(bugget)+','+str(acc)+'\n'
        print(str(bugget)+','+str(acc)+'\n')
    with open('DP_Cifar10','w') as f:
        f.write(content)
        f.flush()

def draw_DP_cifar10():
    for bugget in [0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.8]:
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        # train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
        test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)
        noise = np.random.laplace(loc=0, scale=bugget, size=(10000, 32, 32, 3))

        coin = np.random.random(size=(10000, 32, 32, 3))
        flip = np.where(coin < 1, 1, 0)  # 0.6的概率添加噪声
        noise = noise * flip

        test_x += noise
        test_x = np.clip(test_x, a_min=0, a_max=1)
        plt.imshow(test_x[11])
        plt.title(str(bugget))
        plt.show()

def cifar10():
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
    test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)
    start_test_example_num = 10
    end_test_example_num = 20
    model = keras.models.load_model('DP_model/DP_cifar10_0.7158_mean=0.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0001)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x,test_y)
    # model.summary() 70.91 67 70.52 73 71.01 60
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    # model.summary()
    feature = model(test_x[start_test_example_num:end_test_example_num])[1].numpy()
    y = model(test_x[start_test_example_num:end_test_example_num])[0].numpy()
    y = get_lable_from_distribution(y)
    feature =(feature * 0.5 + 0.5) +test_x[start_test_example_num:end_test_example_num]
    feature=np.clip(feature, a_min=0, a_max=1)

    for i in range(start_test_example_num,end_test_example_num):
        plt.imshow(test_x[i])
        plt.show()
        plt.imshow(feature[i-start_test_example_num])
        plt.title(str(test_y[i]) + '_' + str(y[i-start_test_example_num]))
        plt.show()

#mean and ver
def cifar10_mean_and_var():
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x = (train_x / 255.0).reshape(-1, 32, 32, 3)
    test_x = (test_x / 255.0).reshape(-1, 32, 32, 3)
    start_test_example_num = 0
    end_test_example_num = 1
    model = keras.models.load_model('paramter analysis/model/vgg19_test.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    # model.evaluate(test_x,test_y)
    # model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature_my_dense3 = model.get_layer('my_dense3').output
    feature_my_dense4 = model.get_layer('my_dense4').output
    model = keras.Model(inputs=inputs, outputs=[y, feature_my_dense3,feature_my_dense4])
    # model.summary()
    feature = model(test_x[start_test_example_num:end_test_example_num])
    mean=feature[1].numpy().reshape(300)[:15]
    var=np.exp(feature[2].numpy().reshape(300)[:100])
    # print(mean)
    print(var)
    # img=mean+np.random.standard_normal(mean.shape)*var
    # print(mean[:50])
    # print(np.exp(var[:50] * 0.5))
    # print(img[:50])

def mnist():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = (train_x / 255.0).reshape(-1, 28, 28)
    test_x = (test_x / 255.0).reshape(-1, 28, 28)
    start_test_example_num = 130
    end_test_example_num = 150
    model = keras.models.load_model('model/mnist_combined_SP.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x,test_y)
    model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    model.summary()
    feature = model(test_x[start_test_example_num:end_test_example_num])[1].numpy()
    y = model(test_x[start_test_example_num:end_test_example_num])[0].numpy()
    y = get_lable_from_distribution(y)
    feature = (feature.reshape(-1,28,28) * 0.5 + 0.5)
    for i in range(start_test_example_num,end_test_example_num):
        plt.imshow(test_x[i])
        plt.show()
        plt.imshow(feature[i-start_test_example_num])
        plt.title(str(test_y[i]) + '_' + str(y[i-start_test_example_num]))
        plt.show()
def Fmnist():
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
    train_x = (train_x / 255.0).reshape(-1, 28, 28)
    test_x = (test_x / 255.0).reshape(-1, 28, 28)
    start_test_example_num = 0
    end_test_example_num = 10
    model = keras.models.load_model('model/fashion_mnist_combined_no_additional_information.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x,test_y)
    model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    model.summary()
    feature = model(test_x[start_test_example_num:end_test_example_num])[1].numpy()
    y = model(test_x[start_test_example_num:end_test_example_num])[0].numpy()
    y = get_lable_from_distribution(y)
    feature = (feature.reshape(-1,28,28) * 0.5 + 0.5)
    for i in range(start_test_example_num,end_test_example_num):
        plt.imshow(test_x[i])
        plt.show()
        plt.imshow(feature[i-start_test_example_num])
        plt.title(str(test_y[i]) + '_' + str(y[i-start_test_example_num]))
        plt.show()

def Fmnist_feature_compare():
    #加载noSP模型
    (train_x, train_y), (test_x, test_y) = keras.datasets.fashion_mnist.load_data()
    train_x = (train_x / 255.0).reshape(-1, 28, 28)
    test_x = (test_x / 255.0).reshape(-1, 28, 28)
    model = keras.models.load_model('model/fashion_mnist_noSP.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x, test_y)
    # model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    # model.summary()


    #加载SP模型
    SP_model = keras.models.load_model('model/fashion_mnist_combined_no_additional_information.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    SP_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    SP_model.evaluate(test_x, test_y)
    # SP_model.summary()
    inputs = SP_model.inputs
    y = SP_model(inputs)
    feature = SP_model.get_layer('my_conv15').output
    SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
    # SP_model.summary()

    #加载分类器用于提取原特征比对
    classifer = keras.models.load_model('model/fashion_mnist_classifier.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    classifer.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    classifer.evaluate(test_x, test_y)
    # classifer.summary()
    inputs = classifer.inputs
    y = classifer(inputs)
    middle_feature = classifer.get_layer('dense').output
    classifer = keras.Model(inputs=inputs, outputs=[y,middle_feature])
    # classifer.summary()

    #特征比对
    result='SP,noSP\n'
    for i in range(1,1000):
        image_noSP = model(test_x[i].reshape(-1, 28, 28))[1].numpy()
        image_noSP = (image_noSP.reshape(-1, 28, 28) * 0.5 + 0.5)
        image_SP = SP_model(test_x[i].reshape(-1, 28, 28))[1].numpy()
        image_SP = (image_SP.reshape(-1, 28, 28) * 0.5 + 0.5)

        feature_noSP=classifer(image_noSP)[1].numpy()
        feature_SP=classifer(image_SP)[1].numpy()
        original_feature=classifer(test_x[i].reshape(-1, 28, 28))[1].numpy()
        result+=str(tf.reduce_mean((original_feature-feature_SP)**2).numpy())+','+str(tf.reduce_mean((original_feature-feature_noSP)**2).numpy())+'\n'
        print('\r '+str(i),end='')
    with open('record/fmnist_feature_comparse.csv','w') as f:
        f.write(result)
        f.flush()
    print('done')

def mnist_feature_compare():
    #加载noSP模型
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
    train_x = (train_x / 255.0).reshape(-1, 28, 28)
    test_x = (test_x / 255.0).reshape(-1, 28, 28)
    model = keras.models.load_model('model/mnist_combined_noSP.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x, test_y)
    # model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    # model.summary()


    #加载SP模型
    SP_model = keras.models.load_model('model/mnist_combined_SP.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    SP_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    SP_model.evaluate(test_x, test_y)
    # SP_model.summary()
    inputs = SP_model.inputs
    y = SP_model(inputs)
    feature = SP_model.get_layer('my_conv15').output
    SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
    # SP_model.summary()

    #加载分类器用于提取原特征比对
    classifer = keras.models.load_model('model/mnist_classifier.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    classifer.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    classifer.evaluate(test_x, test_y)
    # classifer.summary()
    inputs = classifer.inputs
    y = classifer(inputs)
    middle_feature = classifer.get_layer('dense').output
    classifer = keras.Model(inputs=inputs, outputs=[y,middle_feature])
    # classifer.summary()

    #特征比对
    result='SP,noSP\n'
    for i in range(1,1000):
        image_noSP = model(test_x[i].reshape(-1, 28, 28))[1].numpy()
        image_noSP = (image_noSP.reshape(-1, 28, 28) * 0.5 + 0.5)
        image_SP = SP_model(test_x[i].reshape(-1, 28, 28))[1].numpy()
        image_SP = (image_SP.reshape(-1, 28, 28) * 0.5 + 0.5)

        feature_noSP=classifer(image_noSP)[1].numpy()
        feature_SP=classifer(image_SP)[1].numpy()
        original_feature=classifer(test_x[i].reshape(-1, 28, 28))[1].numpy()
        result+=str(tf.reduce_mean((original_feature-feature_SP)**2).numpy())+','+str(tf.reduce_mean((original_feature-feature_noSP)**2).numpy())+'\n'
        print('\r '+str(i),end='')
    with open('record/mnist_feature_comparse.csv','w') as f:
        f.write(result)
        f.flush()
    print('done')

def cifar10_feature_compare():
    #加载noSP模型
    (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
    train_x = (train_x / 255.0).reshape(-1, 32, 32,3)
    test_x = (test_x / 255.0).reshape(-1, 32, 32,3)
    model = keras.models.load_model('model/cifar_SP.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    model.evaluate(test_x, test_y)
    # model.summary()
    inputs = model.inputs
    y = model(inputs)
    feature = model.get_layer('my_conv15').output
    model = keras.Model(inputs=inputs, outputs=[y, feature])
    # model.summary()


    #加载SP模型
    SP_model = keras.models.load_model('model/cifar10_combined.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    opt = keras.optimizers.Adam(lr=0.0002)
    SCC = keras.losses.SparseCategoricalCrossentropy()
    SP_model.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    SP_model.evaluate(test_x, test_y)
    # SP_model.summary()
    inputs = SP_model.inputs
    y = SP_model(inputs)
    feature = SP_model.get_layer('my_conv15').output
    SP_model = keras.Model(inputs=inputs, outputs=[y, feature])
    # SP_model.summary()

    #加载分类器用于提取原特征比对
    classifer = keras.models.load_model('model/cifar10_classifier_0.7553.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
    classifer.compile(optimizer=opt, loss=SCC, metrics=['accuracy'])
    classifer.evaluate(test_x, test_y)
    # classifer.summary()
    inputs = classifer.inputs
    y = classifer(inputs)
    middle_feature = classifer.get_layer('dense').output
    classifer = keras.Model(inputs=inputs, outputs=[y,middle_feature])
    classifer.summary()

    #特征比对
    result='SP,noSP\n'
    for i in range(1,1000):
        image_noSP = model(test_x[i].reshape(-1, 32, 32,3))[1].numpy()
        image_noSP = (image_noSP.reshape(-1, 32, 32,3) * 0.5 + 0.5)
        image_SP = SP_model(test_x[i].reshape(-1, 32, 32,3))[1].numpy()
        image_SP = (image_SP.reshape(-1, 32, 32,3) * 0.5 + 0.5)

        feature_noSP=classifer(image_noSP)[1].numpy()
        feature_SP=classifer(image_SP)[1].numpy()
        original_feature=classifer(test_x[i].reshape(-1, 32, 32,3))[1].numpy()
        result+=str(tf.reduce_mean((original_feature-feature_SP)**2).numpy())+','+str(tf.reduce_mean((original_feature-feature_noSP)**2).numpy())+'\n'
        print('\r '+str(i),end='')
    with open('record/cifar10_feature_comparse.csv','w') as f:
        f.write(result)
        f.flush()
    print('done')

def estimation():
    content = 'real_mean,est_mean,real_S,est_S\n'
    for i in range(100):
        y = np.random.normal(loc=2.4536426, scale=1.0286165, size=2)
        mean = np.mean(y)
        S = np.sqrt(np.var(y))
        content+='2.4536426,'+str(mean)+',1.0286165,'+str(S)+'\n'
    with open('estimation5.csv','w') as f:
        f.write(content)
        f.flush()


if __name__ == '__main__':
    # Fmnist()
    # Fmnist_feature_compare()
    # mnist_feature_compare()
    # cifar10_feature_compare()
    # cifar10_mean_and_var()
    # estimation()
    # DP_cifar10()
    cifar10()
    #draw_DP_cifar10()

