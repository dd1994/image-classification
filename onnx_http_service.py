from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from util.transform import ToRGBTransform
from onnx_predict import load_index_to_species_id_from_csv, onnx_inference, input_size
import os

app = Flask(__name__)

# 预加载类别映射和 ONNX 路径
csv_file_path = 'index_to_species_id.csv'
onnx_file_path = 'last.onnx'
index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)

def prepare_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    transform = transforms.Compose([
        ToRGBTransform(),
        transforms.Resize(int(input_size * 1.2)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

@app.route('/')
def index():
    return send_from_directory('.', 'frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image_bytes = file.read()
        image_tensor = prepare_image(image_bytes)
        cpu_image_tensor = image_tensor.cpu()
        result = onnx_inference(onnx_file_path, cpu_image_tensor, index_to_species_id, top_k=3)
        # 格式化输出
        print(f"\n推理耗时: {result['inference_time_ms']:.2f} ms\nTop 3 预测结果:")
        for idx, item in enumerate(result['results'], 1):
            print(f"  {idx}. {item['species']} (概率: {item['probability'] * 100:.2f}%)")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)