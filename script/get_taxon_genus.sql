WITH RECURSIVE split_ancestry AS (
    -- 初始：从 temp_taxon 出发，获取每个物种的 ancestry 并拆分第一个祖先
    SELECT
        tt.taxonID AS species_id,
        1 AS depth,                                      -- 祖先深度，1 表示最远的祖先
        CAST(SUBSTRING_INDEX(t.ancestry, '/', 1) AS UNSIGNED) AS ancestor_id,
        SUBSTRING(t.ancestry, LENGTH(SUBSTRING_INDEX(t.ancestry, '/', 1)) + 2) AS remaining_ancestry
    FROM temp_taxon tt
    LEFT JOIN taxon20251111 t ON tt.taxonID = t.id
    WHERE t.ancestry IS NOT NULL AND t.ancestry != ''   -- 仅处理有 ancestry 的物种

    UNION ALL

    -- 递归：继续拆分剩余的祖先路径
    SELECT
        sa.species_id,
        sa.depth + 1,
        CAST(SUBSTRING_INDEX(sa.remaining_ancestry, '/', 1) AS UNSIGNED),
        SUBSTRING(sa.remaining_ancestry, LENGTH(SUBSTRING_INDEX(sa.remaining_ancestry, '/', 1)) + 2)
    FROM split_ancestry sa
    WHERE sa.remaining_ancestry != ''
),
-- 关联 taxon20251111 获取每个祖先的 rank，并只保留 genus 层级
ancestors_with_rank AS (
    SELECT
        sa.species_id,
        sa.ancestor_id,
        sa.depth,
        t.rank
    FROM split_ancestry sa
    JOIN taxon20251111 t ON sa.ancestor_id = t.id
    WHERE t.rank = 'genus'   -- 仅筛选属
),
-- 对每个物种，选取深度最大的 genus 祖先（即最接近该物种的属）
genus_for_species AS (
    SELECT
        species_id,
        ancestor_id AS genus_id
    FROM (
        SELECT
            species_id,
            ancestor_id,
            ROW_NUMBER() OVER (PARTITION BY species_id ORDER BY depth DESC) AS rn
        FROM ancestors_with_rank
    ) ranked
    WHERE rn = 1
)
-- 最终结果：所有 temp_taxon 中的物种，即使没有 genus 也保留（此时 genus_id 为 NULL）
SELECT
    tt.taxonID  as id,
    gfs.genus_id as genus
FROM temp_taxon tt
LEFT JOIN genus_for_species gfs ON tt.taxonID = gfs.species_id;