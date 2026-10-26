-- Port-pair + progress-bucket historical remaining hours.
-- For each (port_pair, progress_bucket) slot, computes the median/mean/count of
-- remaining voyage hours, estimated from historical completed legs.
--
-- progress_bucket = FLOOR(progress_pct * 10) / 10.0 (i.e. 0.0, 0.1, ..., 0.9)
-- At inference time: bucket = floor(progress_pct * 10) / 10.0

WITH leg_data AS (
    SELECT
        CONCAT(ss.START_PORT_ID, '_', ss.END_PORT_ID) AS port_pair,
        ss.ESTIMATED_DISTANCE / ss.ESTIMATED_SPEED    AS planned_h,
        DATEDIFF('second',
            ss.TIME_STARTED,
            ss.TIME_COMPLETED) / 3600.0               AS actual_h
    FROM EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE ss
    JOIN EDPPROD.EDP_SOURCE.WEBATDS_SAILINGASSET sa
        ON sa.RELATED_SAILING_ID = ss.ID
    JOIN EDPPROD.EDP_SOURCE.WEBATDS_TUG t
        ON t.COMPANYASSET_PTR_ID = sa.ASSET_ID
    WHERE ss.IS_DELETED = FALSE
      AND ss.IS_CURRENT  = TRUE
      AND ss.TIME_COMPLETED IS NOT NULL
      AND ss.TIME_STARTED   IS NOT NULL
      AND ss.ESTIMATED_DISTANCE > 0
      AND ss.ESTIMATED_SPEED    > 0
      AND t.MMSI IS NOT NULL
      AND ss.TIME_STARTED < ss.TIME_COMPLETED
      AND DATEDIFF('second', ss.TIME_STARTED, ss.TIME_COMPLETED) / 3600.0
            BETWEEN 0.5 AND (ss.ESTIMATED_DISTANCE / ss.ESTIMATED_SPEED * 10)
),

-- Ten progress-bucket midpoints: 0.05, 0.15, ..., 0.95
-- Bucket floor = midpoint - 0.05 (0.0, 0.1, ..., 0.9)
progress_buckets AS (
    SELECT column1 AS bucket_midpoint,
           column1 - 0.05 AS progress_bucket
    FROM VALUES (0.05), (0.15), (0.25), (0.35), (0.45),
                (0.55), (0.65), (0.75), (0.85), (0.95)
),

expanded AS (
    SELECT
        l.port_pair,
        b.progress_bucket,
        l.actual_h * (1.0 - b.bucket_midpoint) AS remaining_h_at_bucket
    FROM leg_data l
    CROSS JOIN progress_buckets b
    WHERE l.actual_h * (1.0 - b.bucket_midpoint) > 0
)

SELECT
    port_pair,
    progress_bucket,
    MEDIAN(remaining_h_at_bucket) AS pp_hist_remaining_median,
    AVG(remaining_h_at_bucket)    AS pp_hist_remaining_mean,
    COUNT(*)                       AS pp_hist_remaining_count
FROM expanded
GROUP BY port_pair, progress_bucket
ORDER BY port_pair, progress_bucket
