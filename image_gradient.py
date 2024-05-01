import cv2
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def reverse(img):
    x = img.reshape(-1)
    temp = []
    for i, x_ in enumerate(x):
        temp.append(255 - x_)
    temp = np.asarray(temp).reshape(32,32,3)
    return temp

def gradient(img,mode=''):
    row, column, channel = img.shape
    img_f = np.copy(img)
    img_f = img_f.astype("float")
    gradient = np.zeros((row, column,channel))
    for c in range(channel):
        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(img_f[x + 1, y,c] - img_f[x, y,c])
                gy = abs(img_f[x, y + 1,c] - img_f[x, y,c])
                gradient[x, y,c] = gx + gy

    sharp = img_f + gradient
    sharp = np.where(sharp < 0, 0, np.where(sharp > 255, 255, sharp))

    gradient = gradient.astype("uint8")
    sharp = sharp.astype("uint8")
    plt.imshow(img)
    plt.title(mode)
    plt.show()
    plt.imshow(gradient)
    plt.title(mode)
    plt.show()
    return gradient


if __name__ == '__main__':
    img=cv2.imread('1.jpg')
    gradient(img)



    # (trainx, trainy), (_, _) = keras.datasets.cifar10.load_data()
    # x = trainx[15]
    # y = reverse(x)
    # noise=np.random.randint(low=0,high=255,size=(32,32,3))
    # z= np.ceil((x + noise) / 2).astype(int)
    # noise2=np.random.randint(low=200,high=255,size=(32,32,3))
    # z2= np.ceil((x + noise2) / 2).astype(int)
    # x_g=gradient(x)
    # y_g=gradient(y, mode='y')
    # z_g=gradient(z, mode='z')
    # z2_g=gradient(z2, mode='z2')
    # print('MSE(x_g,y_g)=%.5f'%(np.mean(x_g)-np.mean(y_g))**2)
    # print('MSE(x_g,z_g)=%.5f'%(np.mean(x_g)-np.mean(z_g))**2)
    # print('MSE(x_g,z2_g)=%.5f'%(np.mean(x_g)-np.mean(z2_g))**2)
    # print('MSE(x,z)=%.5f'%(np.mean(x)-np.mean(z))**2)
    # print('MSE(x,z2)=%.5f'%(np.mean(x)-np.mean(z2))**2)


  