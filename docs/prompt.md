# 对训练集和验证集进行再平衡

我在使用 pytorch 进行物种图像分类训练，训练集的文件夹结构示例如下:

```
train -> Reptilia -> 2019
train -> Amphibians -> 238899
```

train 文件夹下有几个个文件夹表示爬行/两栖/植物/昆虫等大类，每个文件又有大量物种 id 命名的文件夹，文件夹下是每个物种的训练图片。
验证集 valid 文件夹的结构和 train 文件夹结构完全相同。

其中训练集和验证集分割的规则如下：

1. 对于每个物种，10% 的图片作为验证集，如果一个物种的 10% 图片少于 10 张，则取 10 张；如果一个物种的 10% 图片多余 50 张，则只取 50 张）；
2. 取 10% 时，不是随机取，而是按照图片修改时间进行排序，取最老的 10%;

考虑到训练集一直在不停收集图片，所以需要一段脚本，进行再平衡，确保有新的训练图片加入后，运行脚本就能重新分配验证集
（注意：某些大类和物种可能还没有验证集，需要新建文件夹）；

请编写一段 node.js 脚本满足上述需求，一步步思考。

查询现有数据库缺少的分布地信息：

```sql
select * from index_to_species_id_area
         where index_to_species_id_area.SpeciesID in (select species_id from `nature-observation`.index_to_species_id where  index_to_species_id.taxon_name  in (select SUBSTRING_INDEX(latin_name, ' ', 2) from shanghai_plants))
         and area is not null
         and area not like '%上海%'
```
插入省份分布数据

```sql
UPDATE index_to_species_id_area
SET area = CONCAT(area, '、上海')
WHERE area IS NOT NULL
  AND area NOT LIKE '%上海%'
  AND SpeciesID IN (
    SELECT species_id
    FROM `nature-observation`.index_to_species_id
    WHERE taxon_name IN (
      SELECT SUBSTRING_INDEX(latin_name, ' ', 2)
      FROM shanghai_plants
#       WHERE alien_remark != '栽培'
    )
);
```


列出所有省份

```sql
WITH RECURSIVE numbers AS (
    SELECT 0 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100  -- 支持最多101个省份
),
province_data AS (
    SELECT DISTINCT  -- 添加DISTINCT确保处理唯一行
        area
    FROM species_area  -- 替换为您的表名
)
SELECT DISTINCT
    TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(area, '、', n + 1), '、', -1)) AS province
FROM province_data
JOIN numbers
    ON n < LENGTH(area) - LENGTH(REPLACE(area, '、', '')) + 1
WHERE TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(area, '、', n + 1), '、', -1)) != ''
ORDER BY province;
```

找出 ppbc id 重复的行：

```sql
SELECT *
FROM (
    SELECT
        id,
        ppbcId,
        chineseName,
        count,
        COUNT(*) OVER (PARTITION BY ppbcId) as duplicate_count
    FROM ppbc_count_result
) AS subquery
WHERE duplicate_count > 1
ORDER BY count DESC
```
