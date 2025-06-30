from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import requests
from io import BytesIO
from PIL import Image
import onnxruntime as ort
import time
import csv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np


class InferRequest(BaseModel):
    image_url: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型和配置相关路径
ONNX_MODEL_PATH = "last.onnx"
CSV_LABEL_PATH = "index_to_species_id.csv"
INPUT_SIZE = 448


# NumPy实现的softmax函数
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# 加载标签映射
def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            index, species_id, chinese_name, taxon_name = row
            display_name = taxon_name if chinese_name == 'NULL' else f"{taxon_name} {chinese_name}"
            index_to_species_id[int(index)] = {
                "preferred_common_name": chinese_name,
                "name": taxon_name,
                "id": species_id
            }
    return index_to_species_id


# 初始化加载标签映射
index_to_species_id = load_index_to_species_id_from_csv(CSV_LABEL_PATH)

# 初始化加载 ONNX 模型
ort_session = ort.InferenceSession(ONNX_MODEL_PATH)


# 图像预处理（使用NumPy替代torch）
def preprocess_image(image_url):
    download_start_time = time.time()
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        download_end_time = time.time()
        download_time_ms = (download_end_time - download_start_time) * 1000

        # 使用PIL处理图像
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # 调整大小
        target_resize = int(INPUT_SIZE * 1.2)
        image = image.resize((target_resize, target_resize), Image.BICUBIC)

        # 中心裁剪
        left = (image.width - INPUT_SIZE) / 2
        top = (image.height - INPUT_SIZE) / 2
        right = (image.width + INPUT_SIZE) / 2
        bottom = (image.height + INPUT_SIZE) / 2
        image = image.crop((left, top, right, bottom))

        # 转换为NumPy数组并归一化
        image_array = np.array(image).astype(np.float32) / 255.0

        # 通道顺序调整为CHW
        image_array = np.transpose(image_array, (2, 0, 1))

        # 标准化 (ImageNet均值和标准差)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image_array = (image_array - mean) / std

        # 添加批次维度
        return np.expand_dims(image_array, axis=0), download_time_ms
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像预处理失败: {str(e)}")


# ONNX 推理函数（使用NumPy替代torch）
def onnx_inference(image_tensor, top_k=3):
    input_name = ort_session.get_inputs()[0].name

    # 确保输入数据类型是float32
    if image_tensor.dtype != np.float32:
        image_tensor = image_tensor.astype(np.float32)

    inference_start_time = time.time()
    outputs = ort_session.run(None, {input_name: image_tensor})[0]
    inference_end_time = time.time()
    inference_time_ms = (inference_end_time - inference_start_time) * 1000

    # 计算softmax概率
    probabilities = softmax(outputs)

    # 获取top-k结果
    top_indices = np.argsort(-probabilities, axis=1)[0, :top_k]

    results = []
    for idx in top_indices:
        species = index_to_species_id.get(int(idx), {
            "preferred_common_name": "未知",
            "name": f"未知物种 (ID: {idx})",
            "id": str(idx)
        })
        prob = probabilities[0, idx]
        results.append({
            "species": species,
            "probability": prob * 1.0,
        })

    return {
        "inference_time_ms": round(inference_time_ms, 2),
        "results": results
    }


# 推理接口
@app.post("/infer")
def infer(data: InferRequest):
    if not data.image_url:
        raise HTTPException(status_code=400, detail="缺少图像 URL 参数")

    try:
        image_tensor, download_time_ms = preprocess_image(data.image_url)
        result = onnx_inference(image_tensor)
        result["download_time_ms"] = round(download_time_ms, 2)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")