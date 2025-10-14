import numpy as np
import matplotlib.pylab as plt

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

def function_2(x):
    return x[0] ** 2 + x[1] ** 2

grad1 = numerical_gradient(function_2,np.array([3.0, 4.0]))
print(grad1)
grad2 = numerical_gradient(function_2,np.array([0.0, 2.0]))
print(grad2)

