CREATE TABLE esi_data (
    subject_rank INT,
    institution VARCHAR(255),
    country_region VARCHAR(255),
    web_of_science_documents INT,
    cites INT,
    cites_per_paper FLOAT,
    top_papers INT,
    filter_value VARCHAR(255),
    PRIMARY KEY (filter_value, subject_rank),
    INDEX idx_institution (institution),
    INDEX idx_country_region (country_region),
    INDEX idx_filter_value (filter_value)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;