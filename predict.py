import torch
import torchvision.transforms as transforms
from PIL import Image

from model import SwinV2Model  # 或者您使用的其他模型类
from util.transform import ToRGBTransform  # 添加导入
import csv


def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行
        for row in reader:
            index, species_id, taxon_name, chinese_name = row
            # index_to_species_id[int(index)] = taxon_name + ' ' + chinese_name
            index_to_species_id[int(index)] = taxon_name + ' ' + chinese_name
    return index_to_species_id

input_size = 448

def main():
    torch.cuda.empty_cache()

    # 多GPU环境下清除指定设备的缓存
    # device_id = 0  # GPU编号
    # with torch.cuda.device(device_id):
    #     torch.cuda.empty_cache()

    csv_file_path = 'index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)
    # amp: agkndjwa
    # 加载检查点文件
    checkpoint = torch.load('wandb_logs/identify/svh1u73y/checkpoints/last.ckpt', map_location=torch.device('cuda:0'), weights_only=True)

    # 创建模型实例
    model = SwinV2Model(num_classes = 474)
    # model.load_state_dict(torch.load('./model_reptilia.pth', map_location=torch.device('cuda:0'), weights_only=True))

    # 如果模型是在Lightning中训练的，你可能需要只提取模型状态字典

    state_dict = checkpoint['state_dict']

    # 加载模型权重
    model.load_state_dict(state_dict)

    # 将模型设置为评估模式
    model.eval()

    # 保存模型为.pth文件
    # torch.save(model.state_dict(), 'model_reptilia.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    # 加载本地图片
    img_path = r"D:\image-classification\data\predict\rep\1.jpg" # 替换为您的图片路径
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
        print(f"预测物种: {species_id}, 概率: {probability * 100:.2f}%")  # 显示为百分比

if __name__ == '__main__':
    main() 