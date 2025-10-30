
# optimizer表示“进行最优化的人”

import numpy as np

class SGD:
    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] = params[key] - self.lr * grads[key]
# SGD低效的根本原因是，梯度的方向并没有指向最小值的方向。

class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentun = momentum
        self.v = None   #新变量v，对应物理上的速度。

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key,val in params.item():
                self.v[key] = np.zeros_like(val)
            # v以字典型变量的形式保存与参数结构相同的数据
        for key in params.keys():
            self.v[key] = self.momentun * self.v - self.lr * grads[key]  #物体在梯度方向上受力，在这个力的作用下，物体的速度增加
            params[key] = params[key] + self.v[key]


# AdaGrad会为参数的每个元素适当地调整学习率，与此同时进行学习
# AdaGrad会记录过去所有梯度的平方和。因此，学习越深入，更新的幅度就越小。
class AdaGrad:
    def __init__(self,lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key,val in params.item():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h = self.h + grads[key] * grads[key]
            grads[key] = grads[key] - self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            #微小值1e-7。这是为了防止当self.h[key]中有0时，将0用作除数的情况。
'''
实际上，如果无止境地学习，更新量就会变为 0，完全不再更新。为了改善这个问题，可以使用 RMSProp 方法。RMSProp方法并不是将过去所有的梯度一视同仁地相加，
而是逐渐地遗忘过去的梯度，在做加法运算时将新梯度的信息更多地反映出来。这种操作从专业上讲，称为“指数移动平均”，呈指数函数式地减小过去的梯度的尺度。
'''

# Adam方法的基本思路:
# Momentum参照小球在碗中滚动的物理规则进行移动，
# AdaGrad为参数的每个元素适当地调整更新步伐。
# 将这两者融合起来


# 如果前一层的节点数为n，则初始值使用标准差为1/np.sqrt(n)的分布
# 当激活函数使用ReLU时，一般推荐使用ReLU专用的初始值，当前一层的节点数为n时，He初始值使用标准差为np.sqrt(2/n)的高斯分布

# 用作激活函数的函数最好具有关于原点对称的性质

# 当激活函数使用ReLU时，权重初始值使用He初始值，
# 激活函数为sigmoid或tanh等S型曲线函数时，初始值使用Xavier初始值。
# 这是目前的最佳实践。（好几年之前的书，现在不知道）