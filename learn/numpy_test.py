import numpy as np
import matplotlib.pylab as plt

'''
#矩阵的宽度必须相同，不然就报错
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([10,20])

C = A * B

print(C)


#神奇的numpy小技巧
x = np.array([-1.0, 1.0, 2.0])
y = x > 0    #比较运算符，得到布尔值，也是二分类，然后再将布尔值转为int，最终得到的结果与激活函数所想要得到的结果模样相同
print(y)
y1 = y.astype(np.int64)  #转换数组的类型
print(y1)

#阶跃函数的图形


def step_function(x):
    return np.array(x > 0 , dtype=np.int64)

x =np.array([-5.0, -4.0, -1.0, 0.0, 1.0, 2.0])
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1,1)
plt.show()


#矩阵的乘积，用来实现神经网络的内积
A = np.array([[1,2],[2,3]])
a1 = np.ndim(A)
a2 = np.shape(A)
B = np.array([[5,6],[9,10]])
C = np.dot(A,B)

print(C)
print(a1)
print(a2)

'''

batch_mask = np.random.choice(6000,10)  #从6000个数里面随机抽选10个数