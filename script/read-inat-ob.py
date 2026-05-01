import csv

# 找出东方蝾螈的所有观察
def filter_taxon_id(file_path, target_taxon_id):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file, delimiter='\t')

        for row_index, row in enumerate(csv_reader):
            if row['taxon_id'] == target_taxon_id:
            #     # 在这里处理符合条件的行数据
                print(f"Row {row_index + 1}: {row}")


# 使用示例
file_path = "E:\Downloads\observations.csv\observations.csv"
target_taxon_id = '1449337'
filter_taxon_id(file_path, target_taxon_id)




