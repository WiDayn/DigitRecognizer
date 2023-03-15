# 导入相关的库
import paddle
import numpy as np
from data_process import get_MNIST_dataloader

train_loader, test_loader = get_MNIST_dataloader()

# 定义模型结构
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear


# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv3 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs, label):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = paddle.reshape(x, [x.shape[0], 980])
        x = self.fc(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True


# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 仅优化算法的设置有所差别
def train(model):
    model = MNIST()
    model.train()

    # 可以选择其他优化算法的设置方案（可修改）
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    # 训练epoch（可修改）
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            images = [t[0] for t in data]
            labels = [t[1] for t in data]
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels, dtype="int32")

            # 前向计算的过程
            predicts, acc = model(images, labels)

            # 计算损失，取一个批次样本损失的平均值（可修改）
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            acc.numpy()))

            # 后向传播，更新参数，消除梯度的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')


# 创建模型
model = MNIST()
# 启动训练过程
train(model)

import pandas as pd

df = pd.read_csv("./work/sample_submission.csv")


def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = test_loader

    now_id = 2

    for batch_id, data in enumerate(eval_loader()):
        images = data
        images = paddle.to_tensor(images)
        predicts = model(images, None)
        predicts = np.argmax(predicts, axis=1)
        for res in predicts:
            df.at[now_id, 'Label'] = res
            now_id = now_id + 1

    df.to_csv('new.csv', index=False)


model = MNIST()
evaluation(model)
