create table temp.temp_taxon
(
    taxonID           int          not null
        primary key,
    iconic_taxon_name varchar(64)  null,
    photoCount        int          null
);


create index temp_taxon_iconic_taxon_name_index
    on temp.temp_taxon (iconic_taxon_name);

create index temp_taxon_photoCount_index
    on temp.temp_taxon (photoCount);


create index temp_taxon_taxonID_index
    on temp.temp_taxon (taxonID);


create table inat_filtered_other_observations
(
    taxon_id         int         null,
    observation_uuid varchar(64) not null
        primary key,
    observed_on      varchar(64) null
);

create index filtered_other_observations_observed_on_index
    on inat_filtered_other_observations (observed_on);

create index filtered_other_observations_taxon_id_index
    on inat_filtered_other_observations (taxon_id);

create table inat_filtered_other_photos
(
    photo_id         int         not null,
    observation_uuid varchar(64) not null,
    extension        varchar(64) null,
    observer_id      int         null,
    photo_uuid       varchar(64) not null,
    taxon_id         int         null
);

create index filtered_other_photos_extension_index
    on inat_filtered_other_photos (extension);

create index filtered_other_photos_observation_uuid_index
    on inat_filtered_other_photos (observation_uuid);

create index filtered_other_photos_observer_id_index
    on inat_filtered_other_photos (observer_id);

create index filtered_other_photos_photo_id_index
    on inat_filtered_other_photos (photo_id);

create index filtered_other_photos_photo_uuid_index
    on inat_filtered_other_photos (photo_uuid);

create index filtered_other_photos_taxon_id_index
    on inat_filtered_other_photos (taxon_id);



insert into inat_filtered_other_observations (taxon_id, observation_uuid, observed_on)
select taxon_id, observation_uuid, observed_on from observations where taxon_id in (select taxonID from valid_other_taxon_count_20251111
where iconic_taxon_name in ("Animalia", "Arachnida", "Mollusca")
and taxonID not in (select taxonID from temp_taxon)
and `rank` = 'species')

-- 如果观察数量多余 400，只保留 400 个观察
WITH RankedObservations AS (
    SELECT
        taxon_id,
        observation_uuid,
        observed_on,
        ROW_NUMBER() OVER (PARTITION BY taxon_id ORDER BY RAND()) as rn
    FROM inat_filtered_other_observations
)

DELETE t1 FROM inat_filtered_other_observations t1
JOIN RankedObservations r ON t1.observation_uuid = r.observation_uuid
WHERE r.rn > 900;


insert inat_filtered_other_photos (photo_id, observation_uuid, extension, observer_id, photo_uuid, taxon_id)
SELECT
    p.photo_id,
    p.observation_uuid,
    p.extension,
    p.observer_id,
    p.photo_uuid,
    fio.taxon_id
    -- 你可以在这里选择更多的字段，根据你的需求
FROM
    inat_filtered_other_observations AS fio
INNER JOIN
    photos AS p
ON
    fio.observation_uuid = p.observation_uuid


-- 如果图片数量多余 900，只保留 900
WITH RankedPhotos AS (
    SELECT
        photo_id,
        taxon_id,
        ROW_NUMBER() OVER (PARTITION BY taxon_id) as rn
    FROM inat_filtered_other_photos
)
DELETE t1 FROM inat_filtered_other_photos t1
JOIN RankedPhotos r ON t1.photo_id = r.photo_id
WHERE r.rn > 900;

select p.*, t.iconic_taxon_name from inat_filtered_other_photos as p
inner join valid_other_taxon_count_20251111 as t
on t.taxonID = p.taxon_id