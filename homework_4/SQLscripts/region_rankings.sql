SELECT filter_value, country_region, SUM(top_papers) AS total_top_papers
FROM esi_data
GROUP BY filter_value, country_region
ORDER BY filter_value, total_top_papers DESC;