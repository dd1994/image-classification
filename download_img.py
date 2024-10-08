import json
import os
import requests
import time

total_downloaded = 0

# 读取JSON文件
with open('./download_img_json/Tringa_nebularia.json', 'r') as f:
    data = json.load(f)

# 创建保存图片的目录
save_dir = 'data/custom_scolopacidae_dataset/000000_Tringa_nebularia'
os.makedirs(save_dir, exist_ok=True)

# 提取前10个图片URL并下载
for i, item in enumerate(data['results']):
    if total_downloaded >= 100:
        break
    # 遍历 item['observation_photos']
    for photo in item['observation_photos']:
        id = photo['photo_id']
        url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{id}/medium.jpeg"
        print(url)
        # 下载图片
        response = requests.get(url)
        if response.status_code == 200:
            # 使用photo_id作为文件名
            filename = f"{id}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # 保存图片
            with open(filepath, 'wb') as img_file:
                img_file.write(response.content)
            total_downloaded += 1
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download image {id}")
        time.sleep(2)    

print("Download complete.")