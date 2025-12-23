SELECT
    e.taxon_id AS taxon_id,
    e.photo_id AS photo_id
FROM temp.exist_photo202511 e
INNER JOIN temp.photos20251111 p ON e.photo_id = p.photo_id
INNER JOIN temp.observations20151111 o ON p.observation_uuid = o.observation_uuid
WHERE e.taxon_id != o.taxon_id