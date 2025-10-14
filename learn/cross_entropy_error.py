#交叉熵误差
#one-hot表示：只有正确标签的的索引为1，其余为0
#交叉熵误差的值是由正确解标签所对应的输出结果决定的

import numpy as np

def cross_entropy_error(y,t):
    delta = 1e-7
    e = t * np.log(y + delta)
    E = -np.sum(e)

    return E

y1 = ([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = ([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
t = ([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
error1 = cross_entropy_error(np.array(y1),np.array(t))
error2 = cross_entropy_error(np.array(y2),np.array(t))
print(error1)
print(error2)
