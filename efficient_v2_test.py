import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.datasets import INaturalist
import multiprocessing

# iNaturalist 图像大小通常较大，我们保持 EfficientNetV2 的默认输入大小
input_size = 224

# 定义数据增强和预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载 iNaturalist 数据集
data_dir = './data'  # 数据集的根目录
batch_size = 16  # 减小批量大小以适应更大的数据集
num_epochs = 3
num_classes = 3  # iNaturalist 2021_train_mini 的类别数

def main():
    # 设置设备和并行
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = multiprocessing.cpu_count() // 4
    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")

    # 加载 EfficientNetV2 预训练模型
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # 获取模型的最后一层输出特征数
    num_features = model.classifier[1].in_features

    # 替换模型的分类头，适应 iNaturalist 数据集的类别数
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # 将模型移动到设备上
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载 iNaturalist 数据集
    train_dataset = INaturalist(root=data_dir, version='2021_train_mini', download=False, transform=transform['train'])
    val_dataset = INaturalist(root=data_dir, version='2021_valid', download=False, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        # 每个 epoch 包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
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

                batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                batch_acc = running_corrects.double() / ((batch_idx + 1) * inputs.size(0))
                print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print('Finished Training')

    # 保存模型
    torch.save(model.state_dict(), 'efficientnet_v2_inat_model.pth')

if __name__ == '__main__':
    main()