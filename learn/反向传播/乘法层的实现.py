

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):     #前向传播，接受来自上一层的x和y，然后相乘
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self,dout):   #反向传播，接受来自下一层的单数dout，然后反转x和y跟传递进来的导数相乘，最后的结果是新的导数然后传递给上游
        dx = dout * self.y
        dy = dout * self.x
        return dx,dy
    

'''
用这个乘法层解决苹果和销售税的计算图问题
'''

apple = 100
apple_num = 2
tax = 1.1

mul_apple_player = MulLayer()
mul_tax_player = MulLayer()

apple_price = mul_apple_player.forward(apple,apple_num)
price = mul_tax_player.forward(apple_price,tax)

print(price)

#backward
dprice = 1
dapple_price, dtax = mul_tax_player.backward(dprice)
dapple, dapple_num = mul_apple_player.backward(dapple_price)

print(dapple,dapple_num,dtax)