from paddle import fluid
import pandas as pd
import numpy as np

train_dir = './work/train.csv'
test_dir = './work/test.csv'

data = pd.read_csv(train_dir)
x = np.array(data).astype('float32')
train_num = x.shape[0]
TrainSet_labels, TrainSet_datas = np.hsplit(x, [1])

data = pd.read_csv(test_dir)
x = np.array(data).astype('float32')
test_num = x.shape[0]
TestSet_datas = x


def train_reader():
    for i in range(train_num):
        x = np.reshape(TrainSet_datas[i], [28, 28])
        x = x / 255
        x = x[np.newaxis, :]
        yield x, TrainSet_labels[i]


def test_reader():
    for i in range(test_num):
        x = np.reshape(TestSet_datas[i], [28, 28])
        x = x / 255
        x = x[np.newaxis, :]
        yield x


TrainSet_reader = fluid.io.batch(train_reader, batch_size=100)
TestSet_reader = fluid.io.batch(test_reader, batch_size=100)


def get_MNIST_dataloader():
    return TrainSet_reader, TestSet_reader