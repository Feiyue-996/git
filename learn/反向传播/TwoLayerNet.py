import numpy as np 
from collections import OrderedDict   #OrderedDict是有序字典，“有序”是指它可以记住向字典里添加元素的顺序


class Relu:
    def __init__(self):
        self.mask = None  # 创建了一个布尔掩码（mask）

    def forward(self,x):
        self.mask = (x <= 0)  #对张量 x 中的每个元素进行判断，检查是否小于或等于0,True 表示对应位置的元素 ≤ 0,False 表示对应位置的元素 > 0
        out = x.copy()
        out[self.mask] = 0  #使用布尔掩码进行索引，将所有标记为 True 的位置（即 ≤0 的元素）设置为0

        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        return out
    
    def backward(self,dout):
        dx = dout * (1 - self.out) * self.out
        return dx
    

#权重矩阵和x的乘机并于偏置进行比较
class Affine:
    def __init__(self,w,b):
        self.w =w
        self.b = b 
        self.x = None
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x 
        out = np.dot(x , self.w) + self.b

        return out
    
    def backward(self,dout):
        dx = np.dot(dout , self.w.T)
        self.dw = np.dot(self.x.T , dout)
        self.db = np.sum(dout , asix = 0)

        return dx
    

#soft-with-Loss层
#输出层的softmax函数
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y,t):
    delta = 1e-7
    e = t * np.log(y + delta)
    E = -np.sum(e)

    return E

class SoftWithLoss:
    def __init__(self):
        self.loss = None  #损失
        self.y = None       #softmax的输出  
        self.t = None       #监督数据（one-hot vector)

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout = 1):    #反向传播时，将要传播的值除以批的大小（batch_size）后，传递给前面的层的是单个数据的误差
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    

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


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.1):   
        #初始化权重
        #weight_init_std:权重初始化标准差，就是学习率，小值避免梯度爆炸（超参）
        self.params = {}   #创建一个空字典来存储所有网络参数
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        #生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2']) 
        self.lastlayer = SoftWithLoss()

    # OrderedDict是有序字典，“有序”是指它可以记住向字典里添加元素的顺序,神经网络的正向传播只需按照添加元素的顺序调用各层的forward()方法就可以完成处理，
    # 而反向传播只需要按照相反的顺序调用各层即可

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        loss = self.lastlayer.forward(y,t)
        return loss
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)   #取最大值
        if t.ndim != 1 : t = np.argmax(t , axis=1)   
        # if t.ndim != 1:
        #   t = np.argmax(t , axis=1)  
        # 检查真实标签 t 的维度,
        # 如果 t.ndim == 1：说明已经是类别编号，如 [2, 0, 1],不需要转换
        # 如果 t.ndim != 1：说明是one-hot编码，需要转换
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    #两种计算梯度的方法
    #第一种：数值梯度
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x,t)  #创建损失函数
        # 梯度 ≈ (f(W + ε) - f(W - ε)) / (2ε)
        grads = {}
        grads['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads
    
    #第二种计算方法，反向传播
    def gradient(self, x, t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        # 而反向传播只需要按照相反的顺序调用各层即可
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affinr1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        return grads
'''
值微分的优点是实现简单，因此，一般情况下不太容易出错。而误差反向传播法的实现很复杂，容易出错。所以，经常会比较数值微分的结果和
误差反向传播法的结果，以确认误差反向传播法的实现是否正确。确认数值微分求出的梯度结果和误差反向传播法求出的结果是否一致（严格地讲，是
非常相近）的操作称为梯度确认（gradient check）
'''






        

