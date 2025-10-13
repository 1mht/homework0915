CREATE TABLE esi_data (
    subject_rank INT,
    institution VARCHAR(255),
    country_region VARCHAR(255),
    web_of_science_documents INT,
    cites INT,
    cites_per_paper FLOAT,
    top_papers INT,
    filter_value VARCHAR(255),
    PRIMARY KEY (filter_value, subject_rank)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;