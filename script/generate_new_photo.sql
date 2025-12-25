create table if not exists temp.fungi_count_20251111
(
    taxonID    int          null,
    commonName varchar(256) null,
    taxonName  varchar(256) null,
    `rank`     varchar(64)  null,
    iconic_taxon_name varchar(64)  null,
    photoCount int null
);

truncate table valid_fungi_count_20251111

UPDATE temp.fungi_count_20251111 AS t1
JOIN temp.taxon20251111 AS t2 ON t1.taxonID = t2.id and t2.active = 'true'
SET t1.rank = t2.rank;

UPDATE temp.fungi_count_20251111 AS t1
JOIN temp.taxon20251111 AS t2 ON t1.taxonID = t2.id and t2.active = 'true'
SET t1.taxonName = t2.name;

UPDATE temp.fungi_count_20251111 AS t1
JOIN temp.chinesename20251111 AS t2 ON t1.taxonID = t2.taxon_id
SET t1.commonName = t2.vernacularName;

where valid_fungi_count_20251111.commonName is null

select taxonID from temp.valid_fungi_count_20251111

-- 查看符合条件的记录数量
DELETE v
FROM temp.valid_fungi_count_20251111 v
WHERE
    NOT EXISTS (
        SELECT 1
        FROM temp.sp2000cntaxon2025 s
        WHERE s.species = v.taxonName
    )
    AND
    NOT EXISTS (
        SELECT 1
        FROM temp.chinesename20251111 c
        WHERE c.taxon_id = v.taxonID
    )
    AND
    v.photoCount < 3;

-- 插入匹配的 taxonID
INSERT IGNORE INTO temp.valid_fungi_count_20251111 (taxonID)
SELECT DISTINCT t.id
FROM temp.sp2000cntaxon2025 s
INNER JOIN temp.taxon20251111 t ON t.name = s.species
WHERE t.active = 'true'
  AND NOT EXISTS (
    SELECT 1
    FROM temp.valid_fungi_count_20251111 v
    WHERE v.taxonID = t.id
);

-- 查看匹配数量
SELECT COUNT(*) as 匹配数量
FROM temp.sp2000cntaxon2025 s
INNER JOIN temp.taxon20251111 t ON t.name = s.species
WHERE t.active = 'true';

SELECT COUNT(*) as 匹配数量
FROM temp.sp2000cntaxon2025 s
INNER JOIN temp.chinesename20251111 c ON c.vernacularName = s.speciesChineseName;

-- 插入匹配的 taxonID（来自 chinesename20251111 表的 taxon_id）
INSERT IGNORE INTO temp.valid_fungi_count_20251111 (taxonID)
SELECT DISTINCT c.taxon_id
FROM temp.sp2000cntaxon2025 s
INNER JOIN temp.chinesename20251111 c ON c.vernacularName = s.speciesChineseName
inner JOIN temp.taxon t on c.taxon_id = t.id
WHERE c.taxon_id IS NOT NULL
  AND NOT EXISTS (
    SELECT 1
    FROM temp.valid_fungi_count_20251111 v
    WHERE v.taxonID = c.taxon_id
) and t.active = 'true'

select * from valid_fungi_count_20251111 where taxonID not in (select id from taxon20251111) and photoCount > 10
select * from valid_fungi_count_20251111 where taxonName is null

UPDATE temp.valid_fungi_count_20251111 AS t1
JOIN temp.taxon20251111 AS t2 ON t1.taxonID = t2.id and t2.active = 'true'
SET t1.rank = t2.rank;

UPDATE temp.valid_fungi_count_20251111 AS t1
JOIN temp.taxon20251111 AS t2 ON t1.taxonID = t2.id and t2.active = 'true'
SET t1.taxonName = t2.name;

UPDATE temp.valid_fungi_count_20251111 AS t1
JOIN temp.chinesename20251111 AS t2 ON t1.taxonID = t2.taxon_id
SET t1.commonName = t2.vernacularName;

ALTER TABLE temp.photos20251111
ADD COLUMN taxon_id INT NULL AFTER observation_uuid;

ALTER TABLE temp.photos20251111
ADD INDEX idx_taxon_id (taxon_id);


select count(*) from photos20251111

SELECT
    e.taxon_id AS taxon_id,
    e.photo_id AS photo_id
FROM temp.exist_photo202511 e
INNER JOIN temp.photos20251111 p ON e.photo_id = p.photo_id
INNER JOIN temp.observations20151111 o ON p.observation_uuid = o.observation_uuid
WHERE e.taxon_id != o.taxon_id
limit 10

