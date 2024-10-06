import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights

# CIFAR-10 图像是 32x32，所以我们需要调整图像大小以适应 EfficientNetV2 的输入
input_size = 224

# 定义数据增强和预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),  # 随机裁剪并调整为 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和方差进行归一化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),  # 中心裁剪为 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载 CIFAR-10 数据集
data_dir = './data'  # 数据集的根目录
batch_size = 128
num_epochs = 10
num_classes = 10



def main():
        # 加载 EfficientNetV2 预训练模型
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # 获取模型的最后一层输出特征数
    num_features = model.classifier[1].in_features

    # 替换模型的分类头，适应 CIFAR-10 数据集的类别数（10 类）
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # 将模型移动到 GPU 上（如果有）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 冻结前面的卷积层（可选）
    for param in model.features.parameters():
        param.requires_grad = False
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform['train'])
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                dataloader = train_loader
            else:
                model.eval()  # 设置模型为验证模式
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 反向传播和优化
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # 打印每个 batch 的损失和准确率
                batch_loss = loss.item()
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    print('Finished Training')
if __name__ == '__main__':
    main()