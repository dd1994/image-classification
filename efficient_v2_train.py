import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.datasets import INaturalist
from torch.optim.lr_scheduler import CosineAnnealingLR

NUM_CLASSES = 10

# 输入图像大小
INPUT_SIZE = 224

# 数据集的根目录
DATA_DIR = './data'

# 批量大小
BATCH_SIZE = 32

# 训练轮数
NUM_EPOCHS = 5

# 数据加载的进程数
NUM_WORKERS = 3

# 学习率
LR = 0.001


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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    # optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


    # 加载 iNaturalist 数据集
    train_dataset = INaturalist(root=DATA_DIR, version='2021_train_mini', download=False, transform=transform['train'])
    val_dataset = INaturalist(root=DATA_DIR, version='2021_valid', download=False, transform=transform['val'])

     # 只使用前10个类别的数据
    def filter_dataset(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label < NUM_CLASSES]
        return torch.utils.data.Subset(dataset, indices)

    train_dataset = filter_dataset(train_dataset)
    val_dataset = filter_dataset(val_dataset)

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
        
        scheduler.step()
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
    val_top1_corrects = 0
    val_top3_corrects = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Compute top-1 and top-3 accuracy
            acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
            val_top1_corrects += acc1.item() * inputs.size(0) / 100
            val_top3_corrects += acc3.item() * inputs.size(0) / 100

            val_loss += loss.item() * inputs.size(0)

            batch_loss = val_loss / ((batch_idx + 1) * inputs.size(0))
            batch_top1_acc = val_top1_corrects / ((batch_idx + 1) * inputs.size(0))
            batch_top3_acc = val_top3_corrects / ((batch_idx + 1) * inputs.size(0))
            print(f"Val Batch {batch_idx + 1}/{len(val_loader)}, Loss: {batch_loss:.4f}, "
                  f"Top-1 Acc: {batch_top1_acc:.4f}, Top-3 Acc: {batch_top3_acc:.4f}")

    epoch_val_loss = val_loss / len(val_loader.dataset)
    epoch_val_top1_acc = val_top1_corrects / len(val_loader.dataset)
    epoch_val_top3_acc = val_top3_corrects / len(val_loader.dataset)
    print(f"Val Loss: {epoch_val_loss:.4f}, Top-1 Acc: {epoch_val_top1_acc:.4f}, "
                   f"Top-3 Acc: {epoch_val_top3_acc:.4f}")


    # 保存模型
    torch.save(model.state_dict(), 'efficientnet_v2_inat_model.pth')

if __name__ == '__main__':
    main()