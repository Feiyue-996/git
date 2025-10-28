import numpy as np

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
    



