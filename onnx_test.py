import onnxruntime as ort
import time
import numpy as np
from PIL import Image
import csv

input_size = 448


def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头行
        for row in reader:
            index, species_id, chinese_name, taxon_name = row
            display_name = taxon_name if chinese_name == 'NULL' else f"{taxon_name} {chinese_name}"
            index_to_species_id[int(index)] = display_name
    return index_to_species_id


def softmax(x):
    """NumPy实现的softmax函数"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def preprocess_image(image_path):
    """使用Pillow和NumPy替代torchvision的预处理"""
    # 读取图像并转换为RGB
    image = Image.open(image_path).convert('RGB')

    # 调整大小
    target_resize = int(input_size * 1.2)
    image = image.resize((target_resize, target_resize), Image.BICUBIC)

    # 中心裁剪
    left = (image.width - input_size) / 2
    top = (image.height - input_size) / 2
    right = (image.width + input_size) / 2
    bottom = (image.height + input_size) / 2
    image = image.crop((left, top, right, bottom))

    # 转换为NumPy数组并归一化
    image_array = np.array(image).astype(np.float32) / 255.0  # 确保使用float32

    # 通道顺序调整为CHW
    image_array = np.transpose(image_array, (2, 0, 1))

    # 标准化 (ImageNet均值和标准差)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image_array = (image_array - mean) / std

    # 添加批次维度
    return np.expand_dims(image_array, axis=0)


def onnx_inference(onnx_file_path, image_array, index_to_species_id, top_k=3):
    """使用ONNX运行时进行推理"""
    ort_session = ort.InferenceSession(onnx_file_path)
    input_name = ort_session.get_inputs()[0].name

    # 确保输入数据类型是float32
    if image_array.dtype != np.float32:
        image_array = image_array.astype(np.float32)

    start_time = time.time()
    outputs = ort_session.run(None, {input_name: image_array})[0]
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # 毫秒

    # 计算softmax概率
    probabilities = softmax(outputs)

    # 获取top-k结果
    top_indices = np.argsort(-probabilities, axis=1)[0, :top_k]

    results = []
    for idx in top_indices:
        species_id = index_to_species_id[idx]
        probability = probabilities[0, idx]
        results.append({
            'species': species_id,
            'probability': float(probability),  # 转换为Python float
            'class_index': int(idx)  # 转换为Python int
        })

    return {
        'inference_time_ms': inference_time,
        'results': results
    }


def main():
    csv_file_path = 'index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)
    onnx_file_path = "last.onnx"
    img_path = "1.jpg"
    top_k = 3

    # 预处理图像
    image_array = preprocess_image(img_path)

    # 使用ONNX进行推理
    result = onnx_inference(onnx_file_path, image_array, index_to_species_id, top_k)

    # 打印结果
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    for idx, item in enumerate(result['results'], 1):
        print(f"Top-{idx}: {item['species']} (概率: {item['probability'] * 100:.2f}%)")


if __name__ == '__main__':
    main()