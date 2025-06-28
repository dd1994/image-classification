from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse
import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import time
import csv
from pydantic import BaseModel

class InferRequest(BaseModel):
    image_url: str

app = FastAPI()

# 模型和配置相关路径
ONNX_MODEL_PATH = "last.onnx"
CSV_LABEL_PATH = "index_to_species_id.csv"
INPUT_SIZE = 448

# 自定义 ToRGB 转换
class ToRGBTransform:
    def __call__(self, img):
        return img.convert("RGB")

# 加载标签映射
def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            index, species_id, chinese_name, taxon_name = row
            display_name = taxon_name if chinese_name == 'NULL' else f"{taxon_name} {chinese_name}"
            index_to_species_id[int(index)] = display_name
    return index_to_species_id

index_to_species_id = load_index_to_species_id_from_csv(CSV_LABEL_PATH)

# 图像预处理
def preprocess_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.Compose([
            ToRGBTransform(),
            transforms.Resize(int(INPUT_SIZE * 1.2)),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).cpu().numpy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像预处理失败: {str(e)}")

# ONNX 推理函数
def onnx_inference(image_tensor, top_k=3):
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
    ort_inputs = {ort_session.get_inputs()[0].name: image_tensor}
    start_time = time.time()
    ort_outputs = ort_session.run(None, ort_inputs)
    inference_time = (time.time() - start_time) * 1000  # 毫秒
    outputs = torch.tensor(ort_outputs[0])
    probabilities = torch.softmax(outputs, dim=1)
    top_probs, top_classes = torch.topk(probabilities, top_k)
    results = []
    for i in range(top_k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id.get(class_index, "未知类别")
        prob = top_probs[0][i].item()
        results.append({
            "species": species_id,
            "probability": round(prob * 100, 2),
            "class_index": class_index
        })
    return {
        "inference_time_ms": round(inference_time, 2),
        "results": results
    }

# 推理接口
@app.post("/infer")
def infer(data: InferRequest):
    if not data.image_url:
        raise HTTPException(status_code=400, detail="缺少图像 URL 参数")

    try:
        image_tensor = preprocess_image(data.image_url)
        result = onnx_inference(image_tensor)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")