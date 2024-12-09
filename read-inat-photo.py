import csv


def filter_observation_uuid(file_path, target_observation_uuid):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file, delimiter='\t')

        for row_index, row in enumerate(csv_reader):
            if row['observation_uuid'] == target_observation_uuid:
                #     # 在这里处理符合条件的行数据
                print(f"Row {row_index + 1}: {row}")


# 使用示例
file_path = "E:\Downloads\photos.csv\photos.csv"
# 使用示例
target_observation_uuid = '7ed460d7-bf8d-4924-b07c-9573b6174ee3'
filter_observation_uuid(file_path, target_observation_uuid)



