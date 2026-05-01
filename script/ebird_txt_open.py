"""
GLOBAL UNIQUE IDENTIFIER    全局唯一标识
LAST EDITED DATE    最后编辑日期
TAXONOMIC ORDER     分类顺序
CATEGORY        类别，属or种
TAXON CONCEPT ID    分类概念ID
COMMON NAME     通用名称
SCIENTIFIC NAME     拉丁学名。***
SUBSPECIES COMMON NAME      亚种通用名称
SUBSPECIES SCIENTIFIC NAME      亚种通用学名***
EXOTIC CODE     外来物种代码
OBSERVATION COUNT   观察次数
BREEDING CODE   育种代码
BREEDING CATEGORY   繁殖类别
BEHAVIOR CODE   行为代码
AGE/SEX     年龄/性别
COUNTRY     国家
COUNTRY CODE    国家代码，遵循ISO 3166-2
STATE       州、省
STATE CODE
COUNTY      县
COUNTY CODE
IBA CODE
BCR CODE
USFWS CODE
ATLAS BLOCK     地区集区块
LOCALITY    地点
LOCALITY ID
LOCALITY TYPE   地点类型
LATITUDE    维度
LONGITUDE   精度
OBSERVATION DATE    观察日期
TIME OBSERVATIONS STARTED   观测开始时间
OBSERVER ID     观察者ID
SAMPLING EVENT IDENTIFIER       采样时间标识符
PROTOCOL TYPE   协议类型
PROTOCOL CODE
PROJECT CODE    项目代码
DURATION MINUTES    持续时间
EFFORT DISTANCE KM  努力距离
EFFORT AREA HA      努力面积
NUMBER OBSERVERS    观察员人数
ALL SPECIES REPORTED    已报告的所有物种
GROUP IDENTIFIER    组标识符
HAS MEDIA   有媒体***
APPROVED    已批准
REVIEWED    已审核
REASON      原因
TRIP COMMENTS   旅行评论
SPECIES COMMENTS    物种评论
"""
input_file = 'D:\ebd_relNov-2024\ebd_relNov-2024.txt\ebd_relNov-2024.txt'

with open(input_file, 'r', encoding='utf-8') as file:
    # 读取第一行，作为列名
    header = next(file).strip().split('\t')

    for i, line in enumerate(file):
        if i >= 10:
            break
        elements = line.strip().split('\t')

        row_dict = dict(zip(header, [item.strip() for item in elements]))

        print(row_dict)

        # if row_dict['HAS MEDIA'] != '0' :
        #     print(row_dict['TAXONOMIC ORDER'], row_dict['SCIENTIFIC NAME'], row_dict['SUBSPECIES SCIENTIFIC NAME'], row_dict['HAS MEDIA'], SAMPLING EVENT IDENTIFIER)