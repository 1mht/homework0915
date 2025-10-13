ALTER TABLE esi_data MODIFY institution VARCHAR(255);
ALTER TABLE esi_data ADD INDEX idx_institution (institution);

ALTER TABLE esi_data MODIFY country_region VARCHAR(255);
ALTER TABLE esi_data ADD INDEX idx_country_region (country_region);

ALTER TABLE esi_data MODIFY filter_value VARCHAR(255);
ALTER TABLE esi_data ADD INDEX idx_filter_value (filter_value);

-- 详细说明：
-- 1. 在已有表 esi_data 的基础上增加索引。
-- 2. 索引不会影响数据内容和主键，只提升查询效率。
-- 3. 执行后可用 SHOW INDEX FROM esi_data; 查看索引情况。