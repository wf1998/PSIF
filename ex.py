import numpy as np
import torchvision.datasets
from PIL import Image

from torch import nn
from torch.nn import Conv2d, Flatten, Linear
from torch.optim import optimizer
from torch.utils.data import DataLoader

#取样操作
def sampling(z_mean, z_log_var):
    epsilon = np.random.normal(z_mean)
    return z_mean + np.exp(z_log_var * 0.5) * epsilon

encoder_input_shape = (28, 28, 1)
classifier_input_shape = (28, 28, 1)
decoder_flatten = 28 * 28 * 1
latent_dimension = 100
#建立训练集和测试集
train_data = torchvision.datasets.MNIST(root="/mnist.py",train=True,transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root="/mnist.py",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_dataloader = DataLoader(train_data,batch_size=10)
test_dataloader = DataLoader(test_data,batch_size=10)


class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        #特征提取
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same")
        self.conv2 = Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same")
        self.conv3 = Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding="same")
        self.conv4 = Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding="same")
        #降维
        self.flat1 = Flatten()
        #全连接
        self.linear1 = Linear(in_features=1,out_features=300)
        #self.linear2 = Linear(out_features=100)
        #正态分布的均值和方差
        self.linear3 = Linear(latent_dimension,latent_dimension)
        self.linear4 = Linear(latent_dimension,latent_dimension)

        self.linear11 = Linear(latent_dimension, 100)
        # self.dense12 = keras.layers.Dense(200, activation=tf.nn.leaky_relu,name='dense12')
        self.linear13 = Linear(1,300)
        self.linear14 = Linear(1,decoder_flatten)
        #self.reshape = view(target_shape=encoder_input_shape)
        #对抗性重构
        self.conv11 = Conv2d(in_channels=3,out_channels=256, kernel_size=3, padding="same")
        self.conv12 = Conv2d(in_channels=3,out_channels=128, kernel_size=3, padding="same")
        self.conv13 = Conv2d(in_channels=3,out_channels=64, kernel_size=3, padding="same")
        self.conv14 = Conv2d(in_channels=3,out_channels=32, kernel_size=3, padding="same")
        self.conv15 = Conv2d(in_channels=3,out_channels=1, kernel_size=3, padding="same")
        #self.cloud_model = classifier(transition_layer_num=2)

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat1(self.conv4(x))
        x = self.linear1(x)
        # x = self.dense2(x)
        #获取正态分布的均值
        z_mean = self.linear3(x)
        #获取正态分布的方差
        z_log_var = self.linear4(x)
        epsilon = sampling(z_mean, z_log_var)
        #samp = z_mean + tensflow.exp(z_log_var / 2) * epsilon
        # samp=x
        #x = self.liear14(self.linear13(self.linear11(samp)))
        x = self.view(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        feature = self.conv15(x) * 0.5 + 0.5
        feature=0.99*feature+0.01*img
        y = self.cloud_model(feature)
        return y
'''
    def model(self):
        x = (28, 28, 1)
        return Model(inputs=x, outputs=self.call(x))
'''

image_path = "E:\\桌面\\test\\test1.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)
for data in train_dataloader:
        imgs, targets = data
        output = Mnist(imgs)
        print(output)