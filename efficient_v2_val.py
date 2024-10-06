import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import torch.nn as nn

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 3  # iNaturalist 2021_train_mini 的类别数

# 定义图像预处理
input_size = 224
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 加载模型
model = models.efficientnet_v2_s(num_classes)
num_features = model.classifier[1].in_features
# 替换模型的分类头，适应 iNaturalist 数据集的类别数
model.classifier[1] = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load('efficientnet_v2_inat_model.pth', map_location=device))
model.to(device)
model.eval()

# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probabilities.topk(1, dim=1)
    
    return top_class.item(), top_prob.item()

# 指定图片文件夹路径
image_folder = './test_img'

# 支持的图片格式
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
# 使用示例
for filename in os.listdir(image_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(image_folder, filename)
        predicted_class, probability = predict_image(image_path)
        
        print(f"\n图片: {filename}")
        print(f"预测类别: {predicted_class}")
        print(f"预测概率: {probability:.4f}")
