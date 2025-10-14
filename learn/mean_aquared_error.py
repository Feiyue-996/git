#损失函数之一
import numpy as np

def mean_spuared_error(y,t):
    #y是神经网络的输出，t是监督数据，即y是预测值，t是实际值
    #y-t就是预测值到实际值的误差
    e = ((y - t)**2)/2
    E = np.sum(e)

    return E

y1 = ([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = ([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
error1 = mean_spuared_error(np.array(y1),np.array(t))
error2 = mean_spuared_error(np.array(y2),np.array(t))
print(error1)
print(error2)
