-- 现在我有一张表 collection, 有两列：id 表示物种id, photo_count 表示该物种的图片数量，这个表的物种都是到种级别(species)的；
-- 还有一张表 taxon20250130，有一列 id 也是表示物种id(可以和 collection 的物种id关联)，还有一列 rank 表示分类层级，有species/genus/family 等值，
-- 还有一列 ancestry，表示该物种的物种数上的父级，格式示例： 48460/1/2/355675/3/71263/570，其中每个数字都是一个物种 id，每个 id 都有自己的rank/ancestry 等值；
-- 请写一段 SQL，统计表 collection 中有多少科，每个科的图片数量是多少


WITH RECURSIVE family_tree AS (
    SELECT
        c.id AS species_id,
        t.id AS current_taxon_id,
        t.ancestry,
        t.rank
    FROM collection c
    JOIN taxon20250130 t ON c.id = t.id
    UNION ALL
    SELECT
        ft.species_id,
        parent.id AS current_taxon_id,
        parent.ancestry,
        parent.rank
    FROM family_tree ft
    JOIN taxon20250130 parent
        ON parent.id = SUBSTRING_INDEX(ft.ancestry, '/', -1)
    WHERE ft.rank != 'family'
)
SELECT
    ft.current_taxon_id AS family_id,
    SUM(c.photo_count) AS total_photos
FROM family_tree ft
JOIN collection c ON ft.species_id = c.id
WHERE ft.rank = 'family'
GROUP BY ft.current_taxon_id;


-- 现在我有一张 mysql 表 collection, 有两列：id 表示物种id, photo_count 表示该物种的图片数量，这个表的物种都是到种级别(species)的，还有两个字段 genus 和 family，表示该物种的属和科 id，但目前是空的；
-- 还有一张表 taxon20250130，有一列 id 也是表示物种id(可以和 collection 的物种id关联)，还有一列 rank 表示分类层级，有species/genus/family 等值，
-- 还有一列 ancestry，表示该物种的物种数上的父级，格式示例： 48460/1/2/355675/3/71263/570，其中每个数字都是一个物种 id，每个 id 都有自己的rank/ancestry 等值；
-- 请写一段 SQL，补全 collection 表中的 genus 和 family 字段

WITH RECURSIVE split_ancestry AS (
    SELECT
        t.id AS species_id,
        CAST(SUBSTRING_INDEX(t.ancestry, '/', 1) AS UNSIGNED) AS ancestor_id,
        SUBSTRING(t.ancestry, LENGTH(SUBSTRING_INDEX(t.ancestry, '/', 1)) + 2) AS remaining_ancestry
    FROM taxon20250130 t
    WHERE t.rank = 'species'
    UNION ALL
    SELECT
        sa.species_id,
        CAST(SUBSTRING_INDEX(sa.remaining_ancestry, '/', 1) AS UNSIGNED),
        SUBSTRING(sa.remaining_ancestry, LENGTH(SUBSTRING_INDEX(sa.remaining_ancestry, '/', 1)) + 2)
    FROM split_ancestry sa
    WHERE sa.remaining_ancestry != ''
),
ranked_ancestors AS (
    SELECT
        sa.species_id,
        sa.ancestor_id,
        t.rank,
        ROW_NUMBER() OVER (PARTITION BY sa.species_id, t.rank ORDER BY (LENGTH(sa.remaining_ancestry) = 0) DESC) AS rn
    FROM split_ancestry sa
    JOIN taxon20250130 t ON sa.ancestor_id = t.id
    WHERE t.rank IN ('genus', 'family')
)
UPDATE collection c
JOIN (
    SELECT
        species_id,
        MAX(CASE WHEN `rank` = 'genus' AND rn = 1 THEN ancestor_id END) AS genus,
        MAX(CASE WHEN `rank` = 'family' AND rn = 1 THEN ancestor_id END) AS family
    FROM ranked_ancestors
    GROUP BY species_id
) AS subquery
ON c.id = subquery.species_id
SET c.genus = subquery.genus,
    c.family = subquery.family;