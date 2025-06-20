from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from util.transform import ToRGBTransform
from onnx_predict import load_index_to_species_id_from_csv, input_size
import os
import onnxruntime as ort

app = Flask(__name__)

# 预加载类别映射和 ONNX 路径
csv_file_path = 'index_to_species_id.csv'
onnx_file_path = 'last.onnx'
index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)

# 全局只加载一次 ONNX Session
ort_session = ort.InferenceSession(onnx_file_path)

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

def onnx_inference_fast(image_tensor, index_to_species_id, top_k=3):
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
    import time
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # 毫秒
    import torch
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
        result = onnx_inference_fast(cpu_image_tensor, index_to_species_id, top_k=3)
        print(f"\n推理耗时: {result['inference_time_ms']:.2f} ms\nTop 3 预测结果:")
        for idx, item in enumerate(result['results'], 1):
            print(f"  {idx}. {item['species']} (概率: {item['probability'] * 100:.2f}%)")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)