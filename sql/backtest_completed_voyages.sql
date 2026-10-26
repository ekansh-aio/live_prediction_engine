WITH leg_summary AS (
    SELECT
        VOYAGE_ID,
        MIN(SEQUENCE_NUM)   AS first_seq,
        MAX(SEQUENCE_NUM)   AS last_seq,
        MIN(TIME_STARTED)   AS voyage_started_at,
        MAX(TIME_COMPLETED) AS voyage_completed_at,
        COUNT(*)            AS n_legs
    FROM EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE
    WHERE IS_DELETED = FALSE
      AND IS_CURRENT  = TRUE
    GROUP BY VOYAGE_ID
)
SELECT *
FROM leg_summary
WHERE voyage_completed_at >= %(window_start)s::TIMESTAMP_TZ
  AND voyage_completed_at <  %(window_end)s::TIMESTAMP_TZ
  AND voyage_started_at   IS NOT NULL
ORDER BY voyage_started_at
