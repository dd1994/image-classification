import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import INaturalist
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import timm

NUM_CLASSES = 11
INPUT_SIZE = 448
DATA_DIR = './data'
BATCH_SIZE = 8
NUM_EPOCHS = 20
NUM_WORKERS = 3
LR = 0.001
PATIENCE = 3

IN_COLAB = 'COLAB_GPU' in os.environ
if IN_COLAB:
    DATA_DIR = '/content/drive2/MyDrive'
    BATCH_SIZE = 16
    INPUT_SIZE = 448
    NUM_EPOCHS = 20

# 数据增强和预处理
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(int(INPUT_SIZE * 1.2)),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def accuracy(output, target, topk=(1,)):
    """计算 Top-k 准确率"""
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

def main():
    start_time = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 加载数据集
    full_dataset = INaturalist(root=DATA_DIR, version='2019', download=False, transform=transform['train'])

    # 切分训练集、验证集和测试集，比例为 7:1:2
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 添加早停机制
    patience = PATIENCE  # 设置容忍的epoch数量
    best_val_acc = 0.0
    patience_counter = 0


    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
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
        epoch_end_time = time.time()
        epoch_duration = (epoch_end_time - epoch_start_time) / 60
        print(f"Epoch {epoch} duration: {epoch_duration:.2f} minutes")

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

        # 每个 epoch 结束后验证
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
        # 检查是否提升
        if epoch_val_top1_acc > best_val_acc:
            best_val_acc = epoch_val_top1_acc
            patience_counter = 0  # 重置计数器
            print("Validation accuracy improved.")
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy. Patience counter: {patience_counter}/{patience}")

        # 如果达到容忍阈值，则终止训练
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # 所有 epoch 完成后验证测试集
    # print("\nTesting on test set...")
    # model.eval()
    # test_loss = 0.0
    # test_top1_corrects = 0
    # test_top3_corrects = 0

    # with torch.no_grad():
    #     for batch_idx, (inputs, labels) in enumerate(test_loader):
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         acc1, acc3 = accuracy(outputs, labels, topk=(1, 3))
    #         test_top1_corrects += acc1.item() * inputs.size(0) / 100
    #         test_top3_corrects += acc3.item() * inputs.size(0) / 100

    #         test_loss += loss.item() * inputs.size(0)
    #         batch_loss = test_loss / ((batch_idx + 1) * inputs.size(0))
    #         batch_top1_acc = test_top1_corrects / ((batch_idx + 1) * inputs.size(0))
    #         batch_top3_acc = test_top3_corrects / ((batch_idx + 1) * inputs.size(0))

    #         print(f"Test Batch {batch_idx + 1}/{len(test_loader)}, Loss: {batch_loss:.4f}, "
    #               f"Top-1 Acc: {batch_top1_acc:.4f}, Top-3 Acc: {batch_top3_acc:.4f}")

    # test_loss = test_loss / len(test_loader.dataset)
    # test_top1_acc = test_top1_corrects / len(test_loader.dataset)
    # test_top3_acc = test_top3_corrects / len(test_loader.dataset)
    # print(f"Test Loss: {test_loss:.4f}, Top-1 Acc: {test_top1_acc:.4f}, Top-3 Acc: {test_top3_acc:.4f}")

    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    print(f'Finished Training. Total training time: {total_duration:.2f} minutes')

    # 保存模型
    torch.save(model.state_dict(), 'efficientnet_v2_inat_model.pth')

if __name__ == '__main__':
    main()
