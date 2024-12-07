import json
from collections import Counter


def get_inat2019_class_counts(num_classes: int):
        with open('data/medium/train2019.json', 'r') as f:
            data = json.load(f)

        # 提取所有的 category_id
        category_ids = [annotation['category_id'] for annotation in data['annotations']]

        # 统计每个类别的样本数量
        class_counts = Counter(category_ids)

        # 将结果转换为列表，确保按类别 ID 排序
        # 假设类别 ID 从 0 开始，最大类别 ID 为 1000（根据实际情况调整）
        class_counts = [class_counts.get(i, 0) for i in range(num_classes)]
        return class_counts
