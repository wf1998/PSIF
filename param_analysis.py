import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import experiment_mnist
import os

def D_and_I(dimention,information):
    difference = '参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch\n'
    for i in information:
        for d in dimention:
            max_acc, accs = 0, []
            dir = 'record/引入过渡层/new/test/cifar10_test_' + d + '_' + i + '_2.csv'
            if os.path.exists(dir):
                with open(dir) as f:
                    content = f.readline()
                    while content:
                        val = float(content.split(',')[0])
                        accs.append(val)
                        content = f.readline()
                max_acc = max(accs)
                max_index = accs.index(max_acc)
                convergence_epoch = len(accs) - 10
                difference += d + '_' + i + ',' + str((max_acc - 0.7553) * 100 + 2) + ',' + str(
                    max_index) + ',' + str(convergence_epoch) + ','
        difference = difference[:-1] + '\n'
        with open('record/引入过渡层/new/test/D_and_I.csv', 'w') as f:
            f.write(difference)
            f.flush()
    print(difference)


def D_and_L(dimention,layer):
    difference = '参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch\n'
    for d in dimention:
        for l in layer:
            max_acc, accs = 0, []
            dir = 'record/引入过渡层/new/test/cifar10_test_' + d + '_0.02_' + l + '.csv'
            if os.path.exists(dir):
                with open(dir) as f:
                    content = f.readline()
                    while content:
                        val = float(content.split(',')[0])
                        accs.append(val)
                        content = f.readline()
                max_acc = max(accs)
                max_index = accs.index(max_acc)
                convergence_epoch = len(accs) - 10
                difference += d + '_' + l + ',' + str((max_acc - 0.7553) * 100 + 2) + ',' + str(
                    max_index) + ',' + str(convergence_epoch) + ','
        difference = difference[:-1] + '\n'
        with open('record/引入过渡层/new/test/D_and_L.csv', 'w') as f:
            f.write(difference)
            f.flush()
    print(difference)


def L_and_I(layer,information):
    difference = '参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch,参数,准确率差异,达到最有epoch,收敛epoch\n'
    for i in information:
        for l in layer:
            max_acc, accs = 0, []
            dir = 'record/引入过渡层/new/test/cifar10_test_100_'+ i+'_' + l + '.csv'
            if os.path.exists(dir):
                with open(dir) as f:
                    content = f.readline()
                    while content:
                        val = float(content.split(',')[0])
                        accs.append(val)
                        content = f.readline()
                max_acc = max(accs)
                max_index = accs.index(max_acc)
                convergence_epoch = len(accs) - 10
                difference += i + '_' + l + ',' + str((max_acc - 0.7553) * 100 + 2) + ',' + str(
                    max_index) + ',' + str(convergence_epoch) + ','
        difference = difference[:-1] + '\n'
        with open('record/引入过渡层/new/test/L_and_I.csv', 'w') as f:
            f.write(difference)
            f.flush()
    print(difference)


if __name__ == '__main__':
    layer=['1','2','3','4','5']
    dimention=['500','300','100','50','10']
    information=['0.01','0.02','0.03','0.04','0.06','0.08','0.1']
    #维度和信息
    D_and_I(dimention,information)
    #维度和层#cifar10_test_50_0.01_2.csv
    D_and_L(dimention,layer)
    #层和信息
    L_and_I(layer,information)
