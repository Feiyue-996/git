#实现一个简单的神经网络
#一个输入层，两个隐藏层，一个输出层
#两个输入，两个输出

import numpy as np

#定义激活函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#定义神经网络
def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['w2'] = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.3],[0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

#定义前向传播
def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = a3 #最后一层没有激活函数？根据不同的任务选择输出层的激活函数
    #回归问题用恒等函数，分类问题用softmax函数

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network,x)
print(y)


