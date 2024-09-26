import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 定义 CNN 模型，使用 nn.Sequential 使代码更加简洁
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 卷积层
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平层
            nn.Linear(64 * 8 * 8, 512),  # 全连接层
            nn.ReLU(),
            nn.Linear(512, 10)  # 输出层，10 个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 初始化网络
net = SimpleCNN()

# 打印网络结构以查看层次
print(net)

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGD 优化器

# 训练网络
for epoch in range(10):  # 迭代 10 个 epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据
        inputs, labels = data
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播并更新权重
        loss.backward()
        optimizer.step()
        
        # 打印损失值
        running_loss += loss.item()
        if i % 200 == 199:  # 每200个批次打印一次
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

print('Finished Training')

# 测试网络
correct = 0
total = 0
with torch.no_grad():  # 在测试阶段不需要计算梯度
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test images: {100 * correct / total:.2f}%')

# 保存模型
torch.save(net.state_dict(), 'cnn_model.pth')
