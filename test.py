from django.views import View
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

#神经网络的主体
class Net(torch.nn.Module):
    #四个全连接层
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28,64)
        self.fc2 = torch.nn.Linear(64,64)
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,10)

    #前向传播过程
    def forward(self,x):
        x = torch.nn.functional.relu(self.fc1(x))  #全连接线性计算再套一个relu激活函数
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim = 1) #输出层通过softmax进行归一化处理（log_softmax)是为了提高计算的稳定性，在softmax之外又套了一个对数函数
        return x

#用来导入数据
def get_data_loader(is_train):
    #首先定义数据的转换类型（tensor在中文中称为张量，一般用于处理三维以上的数据）
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("",is_train,transform = to_tensor,download = True) #下载MNIST数据集
    return DataLoader(data_set, batch_size = 15, shuffle = True)  #返回数据加载器

#识别正确率
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x ,y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i ,outputs in enumerate(outputs):
                if torch.argmax(outputs) == y[i]:    #argmax:计算一个数列中的最大值的序号，也就是预测概率最高的数字
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

#主函数
def main():
    #导入训练集和测试集
    train_data = get_data_loader(is_train = True)
    test_data = get_data_loader(is_train = False)
    net = Net()   #初始化神经网络

    print("initial accuracy:", evaluate(test_data ,net)) #在训练开始前，打印初始网络的正确率
    #训练神经网络，PyTorch的固定写法
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    for epoch in range(2):
        for (x,y) in train_data:
            net.zero_grad()  #初始化
            output = net.forward(x.view(-1, 28*28)) #正向传播
            loss = torch.nn.functional.nll_loss(output, y) #损失函数，计算误差(nll_lose是一个对数损失函数)
            loss.backward() #反向传播误差
            optimizer.step() #优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))
    #随机抽取三张图像，显示网络的预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3 :
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction:" + str(int(predict)))
    plt.show()

#主函数入口
if __name__ == "__main__":
    main()




