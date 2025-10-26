

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


class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y
        return out
    
    def backward(self,dout):
        dx = dout * 1
        dy = dout * 1
        return dx,dy
    
apple = 100
apple_num = 2
oringe = 150
oringe_num = 3
tax = 1.1

#layer
mul_apple_player = MulLayer()
mul_oringe_player = MulLayer()
add_apple_oringe = AddLayer()
mul_tax_player = MulLayer()

#forward
apple_price = mul_apple_player.forward(apple,apple_num)
oringe_price = mul_oringe_player.forward(oringe,oringe_num)
all_price = add_apple_oringe.forward(apple_price,oringe_price)
price = mul_tax_player.forward(all_price ,tax)

print(price)

#backward
dprice = 1
dall_price, dtax = mul_tax_player.backward(dprice)
dapple_price,doringe_price = add_apple_oringe.backward(dall_price)
dapple,dapple_num = mul_apple_player.backward(dall_price)
doringe,doringe_num = mul_oringe_player.backward(doringe_price)

print(dall_price,dtax,dall_price,doringe_price,doringe,doringe_num,dall_price,dapple,dapple_num)

