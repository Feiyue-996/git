import numpy as np
import matplotlib.pylab as plt

#求损失函数梯度
def numerical_gradient(f , x):
    h = 1e-4 #0.0001
    grad = np.zeros_like(x)  #生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        #计算f（x+h）
        x[idx] = tmp_val + h
        fxh1 = f(x)
        #计算f（x-h）
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val   #还原

    return grad

def gredient_descent(f, init_x, lr = 0.01, step_sum = 100):
    #学习率 = 0.01；梯度法重复100次
    x = init_x

    for i in range(step_sum):
        grad = numerical_gradient(f, x)
        x = x - (lr * grad)

    return x


