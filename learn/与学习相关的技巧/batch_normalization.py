import numpy as np

'''
每次正向传播时，self.mask中都会以False的形式保存要删除的神经元。self.mask会随机生成和x形状相同的数组，并将值比
dropout_ratio大的元素设为True。
反向传播时的行为和ReLU相同。也就是说，正向传播时传递了信号的神经元，
反向传播时按原样传递信号；正向传播时没有传递信号的神经元，反向传播时信号将停在那里。
'''
class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio   #删除比例，大于等于这个比例的呗删掉，小于这个比例的被保存
        self.mask = None   #一个形状跟x一样的掩码

    def forward(self, x, train_flg = True):
        if train_flg:    #训练阶段（train_flg = True）
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio   #生成跟x形状一样的随机数，随机挑选被暂时释放的神经元
            return x * self.mask   
        else:     #测试阶段 (train_flg=False)
            return x * (1 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask