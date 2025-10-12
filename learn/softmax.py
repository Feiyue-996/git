#一般用于分类问题时，最后输出层的激活函数

import numpy as np

def softmax(a):
    c = np.maximum(a - c)    #防止溢出的对策
    a_exp = np.exp(a)
    a_exp_sum = np.sum(a_exp)
    y = a_exp/a_exp_sum
    return y

#可能会出现的问题：溢出
#计算机在处理数时，数值必须在4字节或者8字节的有效数据宽度内

#因为softmax函数并不会改变最后一层隐藏层的输出的大小关系，所以输出层的softmax韩式一般会被省略