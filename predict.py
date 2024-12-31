import torch
import torchvision.transforms as transforms
from PIL import Image

from model import SwinV2Model  # 或者您使用的其他模型类
from util.transform import ToRGBTransform  # 添加导入
import csv


def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行
        for row in reader:
            index, species_id = row
            index_to_species_id[int(index)] = species_id
    return index_to_species_id

input_size = 448

def main():
    csv_file_path = 'index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)
    # 创建模型实例
    model = SwinV2Model.load_from_checkpoint('wandb_logs/identify/dwt4lj4q/checkpoints/swinv2-Amphibians-small-epoch=13-val/acc_top1=0.9831.ckpt')  # 替换为您的检查点路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # 加载本地图片
    img_path = 'data/predict/amp/test.jpg'  # 替换为您的图片路径
    image = Image.open(img_path)

    # 预处理图片
    transform = transforms.Compose([
        ToRGBTransform(),  # 添加 ToRGBTransform
        transforms.Resize(int(input_size * 1.2)),  # 根据训练时的输入大小调整
        transforms.CenterCrop(input_size),  # 中心裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])
    image = transform(image).unsqueeze(0)  # 添加批次维度

    image = image.to(device)

    # 进行预测
    with torch.no_grad():  # 在预测时禁用梯度计算
        outputs = model(image)  # 直接使用模型进行预测

    # 获取 top 3 预测结果及其概率
    top_k = 3
    probabilities = torch.softmax(outputs, dim=1)  # 计算概率
    top_probs, top_classes = torch.topk(probabilities, top_k)  # 获取 top 3 概率和类别

    # 输出 top 3 预测结果和概率
    for i in range(top_k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id[class_index]
        probability = top_probs[0][i].item()
        print(f"预测物种 ID: {species_id}, 概率: {probability:.4f}")

if __name__ == '__main__':
    main() 