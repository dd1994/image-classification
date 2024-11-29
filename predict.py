from pytorch_lightning import Trainer
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import SwinV2Model  # 或者您使用的其他模型类
import json

from util.transform import ToRGBTransform  # 添加导入

input_size = 448

def main():

    # 创建模型实例
    model = SwinV2Model.load_from_checkpoint('wandb_logs/identify/zdyuh8p2/checkpoints/swinv2-inat2021-mini-epoch=12-val/acc_top1=0.8633.ckpt')  # 替换为您的检查点路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # 加载本地图片
    img_path = 'data/predict/2019/2.jpeg'  # 替换为您的图片路径
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
        print(f"预测类别: {top_classes[0][i].item()}, 概率: {top_probs[0][i].item():.4f}")

if __name__ == '__main__':
    main() 