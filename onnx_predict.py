import onnxruntime as ort  # 添加ONNX运行时库
import time  # 添加time模块用于计时

import torch
import torchvision.transforms as transforms
from PIL import Image

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

def onnx_inference(onnx_file_path, image_tensor, index_to_species_id, top_k=3):
    """使用ONNX运行时进行推理，返回结构化结果"""
    ort_session = ort.InferenceSession(onnx_file_path)
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # 毫秒
    outputs = torch.tensor(ort_outputs[0])
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_classes = torch.topk(probabilities, top_k)
    results = []
    for i in range(top_k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id[class_index]
        probability = top_probs[0][i].item()
        results.append({
            'species': species_id,
            'probability': probability,
            'class_index': class_index
        })
    return {
        'inference_time_ms': inference_time,
        'results': results
    }

def main():
    csv_file_path = 'index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    onnx_file_path = "last.onnx"

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


    top_k = 3



    # ================== 添加ONNX推理 ==================
    # 将图像张量移回CPU（ONNX运行时通常在CPU上工作）
    cpu_image_tensor = image_tensor.cpu()

    # 使用ONNX进行推理
    result = onnx_inference(onnx_file_path, cpu_image_tensor, index_to_species_id, top_k)
    for idx, item in enumerate(result['results'], 1):
        print(f"预测结果：{item['species']} (概率: {item['probability'] * 100:.2f}%)")
    # =================================================


if __name__ == '__main__':
    main()