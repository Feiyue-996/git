#Rectifed Linear Unit
#ReLU函数在输出大于0时直接输出该值，在输出小于等于0时直接输出0
import numpy as np
import matplotlib.pylab as plt

def ReLU(x):
    return np.maximum(0,x)

x = np.arange(-2,4,0.1)
y = ReLU(x)
plt.plot(x,y)
plt.ylim(-0.1,4.1)
plt.show()