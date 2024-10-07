import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.datasets import INaturalist


from constant import BATCH_SIZE, DATA_DIR, LR, NUM_CLASSES, INPUT_SIZE, NUM_EPOCHS, NUM_WORKERS

# 定义数据增强和预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载 iNaturalist 数据集

def main():
    start_time = time.time()  # 记录总训练开始时间

    # 设置设备和并行
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载 EfficientNetV2 预训练模型
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # 获取模型的最后一层输出特征数
    num_features = model.classifier[1].in_features

    # 替换模型的分类头，适应 iNaturalist 数据集的类别数
    model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

    # 将模型移动到设备上
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)


    # 加载 iNaturalist 数据集
    train_dataset = INaturalist(root=DATA_DIR, version='2021_train_mini', download=False, transform=transform['train'])
    val_dataset = INaturalist(root=DATA_DIR, version='2021_valid', download=False, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()  # 记录每个 epoch 开始时间
        print(f"Epoch {epoch}/{NUM_EPOCHS - 1}")
        print('-' * 10)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

            batch_loss = train_loss / ((batch_idx + 1) * inputs.size(0))
            batch_acc = train_corrects.double() / ((batch_idx + 1) * inputs.size(0))

    
            print(f"Train Batch {batch_idx + 1}/{len(train_loader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")
        
        
        epoch_end_time = time.time()  # 记录每个 epoch 结束时间
        epoch_duration = (epoch_end_time - epoch_start_time) / 60  # 转换为分钟
        print(f"Epoch {epoch} duration: {epoch_duration:.2f} minutes")
    
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

    end_time = time.time()  # 记录总训练结束时间
    total_duration = (end_time - start_time) / 60  # 转换为分钟
    print(f'Finished Training. Total training time: {total_duration:.2f} minutes')

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            print(outputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

            batch_loss = val_loss / ((batch_idx + 1) * inputs.size(0))
            batch_acc = val_corrects.double() / ((batch_idx + 1) * inputs.size(0))
            print(f"Val Batch {batch_idx + 1}/{len(val_loader)}, Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")
    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
    print(f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")


    # 保存模型
    torch.save(model.state_dict(), 'efficientnet_v2_inat_model.pth')

if __name__ == '__main__':
    main()