import onnx
import onnxruntime as ort  # 添加ONNX运行时库

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import SwinV2Model  # 或者您使用的其他模型类
from util.transform import ToRGBTransform  # 添加导入
import csv

input_size = 448

def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行
        for row in reader:
            index, species_id, chinese_name, taxon_name = row
            # If chinese_name is empty or only contains whitespace, use taxon_name
            display_name = taxon_name if chinese_name == 'NULL' else f"{taxon_name} {chinese_name}"
            index_to_species_id[int(index)] = display_name
    return index_to_species_id

# ... 保留您原有的导入和函数定义 ...

def export_to_onnx(model, device, input_size=448, onnx_file_path="model_reptilia.onnx"):
    """将PyTorch模型导出为ONNX格式"""
    # 创建虚拟输入（与模型预期输入尺寸相同）
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # 设置动态轴（使批次大小可变）
    dynamic_axes = {
        'input': {0: 'batch_size'},  # 批次维度动态
        'output': {0: 'batch_size'}  # 输出批次维度动态
    }

    # 导出模型
    torch.onnx.export(
        model,  # 要导出的模型
        dummy_input,  # 模型输入（虚拟数据）
        onnx_file_path,  # 输出文件路径
        export_params=True,  # 导出模型权重
        opset_version=14,  # ONNX操作集版本
        do_constant_folding=True,  # 优化常量折叠
        input_names=['input'],  # 输入名称
        output_names=['output'],  # 输出名称
        dynamic_axes=dynamic_axes  # 动态维度
    )

    # 验证导出的ONNX模型
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX模型已成功导出到: {onnx_file_path}")
    print(f"模型输入: {onnx_model.graph.input[0]}")
    print(f"模型输出: {onnx_model.graph.output[0]}")

    return onnx_file_path


def onnx_inference(onnx_file_path, image_tensor, index_to_species_id, top_k=3):
    """使用ONNX运行时进行推理"""
    # 创建ONNX运行时会话
    ort_session = ort.InferenceSession(onnx_file_path)

    # 准备输入（注意：ONNX需要numpy数组而不是torch张量）
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}

    # 运行推理
    ort_outputs = ort_session.run(None, ort_inputs)

    # 处理输出
    outputs = torch.tensor(ort_outputs[0])
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_classes = torch.topk(probabilities, top_k)

    # 输出结果
    print("\nONNX推理结果:")
    for i in range(top_k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id[class_index]
        probability = top_probs[0][i].item()
        print(f"预测物种: {species_id}, 概率: {probability * 100:.2f}%")


def main():
    # ... 保留您原有的模型加载代码 ...

    torch.cuda.empty_cache()

    # 多GPU环境下清除指定设备的缓存
    # device_id = 0  # GPU编号
    # with torch.cuda.device(device_id):
    #     torch.cuda.empty_cache()

    csv_file_path = 'index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)
    # amp: agkndjwa
    # 蜘蛛：d22e4crw
    # 加载检查点文件
    checkpoint = torch.load('last.ckpt', map_location=torch.device('cpu'), weights_only=True)

    # 创建模型实例
    model = SwinV2Model(num_classes = 38001)
    # model.load_state_dict(torch.load('./model_reptilia.pth', map_location=torch.device('cuda:0'), weights_only=True))

    # 如果模型是在Lightning中训练的，你可能需要只提取模型状态字典

    state_dict = checkpoint['state_dict']

    # 加载模型权重
    model.load_state_dict(state_dict)

    # 将模型设置为评估模式
    model.eval()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ================== 添加ONNX导出 ==================
    onnx_file_path = "model_reptilia.onnx"
    export_to_onnx(model, device, input_size, onnx_file_path)
    # =================================================

    # 加载并预处理图片（与之前相同）
    img_path = r"1.jpg"
    image = Image.open(img_path)
    transform = transforms.Compose([
        ToRGBTransform(),
        transforms.Resize(int(input_size * 1.2)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 原始PyTorch推理（保留用于比较）
    with torch.no_grad():
        outputs = model(image_tensor)

    top_k = 3
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_classes = torch.topk(probabilities, top_k)

    print("PyTorch推理结果:")
    for i in range(top_k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id[class_index]
        probability = top_probs[0][i].item()
        print(f"预测物种: {species_id}, 概率: {probability * 100:.2f}%")

    # ================== 添加ONNX推理 ==================
    # 将图像张量移回CPU（ONNX运行时通常在CPU上工作）
    cpu_image_tensor = image_tensor.cpu()

    # 使用ONNX进行推理
    onnx_inference(onnx_file_path, cpu_image_tensor, index_to_species_id, top_k)
    # =================================================


if __name__ == '__main__':
    main()