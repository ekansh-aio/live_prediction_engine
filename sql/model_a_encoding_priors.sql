-- Per-route encoding priors for Model A in naive_residual target space.
-- target = SIGN(actual_h - naive_h) * POWER(ABS(actual_h - naive_h), 1/3)
-- where naive_h = ESTIMATED_DISTANCE / ESTIMATED_SPEED + LOCKS_HOURS_INCLUDED.
--
-- Produces six kinds of rows (all using pair_mean/pair_std/... column names):
--   pair          key = "START_PORT_ID_END_PORT_ID"
--   start_port    key = "START_PORT_ID"
--   end_port      key = "END_PORT_ID"
--   pair_svc      key = "START_PORT_ID_END_PORT_ID_SERVICE_TYPE"
--   start_port_svc key = "START_PORT_ID_SERVICE_TYPE"
--   end_port_svc  key = "END_PORT_ID_SERVICE_TYPE"

WITH base AS (
    SELECT
        CONCAT(ss.START_PORT_ID, '_', ss.END_PORT_ID)                       AS port_pair,
        CAST(ss.START_PORT_ID AS VARCHAR)                                    AS start_port_id,
        CAST(ss.END_PORT_ID   AS VARCHAR)                                    AS end_port_id,
        CAST(ss.SERVICE_TYPE  AS VARCHAR)                                    AS service_type,
        DATEDIFF('second', ss.TIME_STARTED, ss.TIME_COMPLETED) / 3600.0     AS actual_h,
        ss.ESTIMATED_DISTANCE / NULLIF(ss.ESTIMATED_SPEED, 0)
            + COALESCE(ss.LOCKS_HOURS_INCLUDED, 0)                           AS naive_h
    FROM EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE ss
    WHERE ss.IS_DELETED       = FALSE
      AND ss.IS_CURRENT       = TRUE
      AND ss.ESTIMATED_DISTANCE > 0
      AND ss.ESTIMATED_SPEED    > 0
      AND ss.TIME_STARTED    IS NOT NULL
      AND ss.TIME_COMPLETED  IS NOT NULL
      AND ss.TIME_STARTED     < ss.TIME_COMPLETED
),
filtered AS (
    SELECT
        port_pair,
        start_port_id,
        end_port_id,
        service_type,
        SIGN(actual_h - naive_h)
            * POWER(ABS(actual_h - naive_h), 1.0 / 3.0)   AS target
    FROM base
    WHERE actual_h >= 0.5
      AND actual_h <= naive_h * 10.0
      AND naive_h  > 0
),
global_stats AS (
    SELECT
        AVG(target)         AS global_mean,
        STDDEV_SAMP(target) AS global_std
    FROM filtered
),

-- kind: pair
pair_agg AS (
    SELECT
        'port_pair'   AS level,
        port_pair     AS key,
        'pair'        AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY port_pair
    HAVING COUNT(*) >= 2
),

-- kind: start_port
sp_agg AS (
    SELECT
        'start_port'  AS level,
        start_port_id AS key,
        'start_port'  AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY start_port_id
    HAVING COUNT(*) >= 2
),

-- kind: end_port
ep_agg AS (
    SELECT
        'end_port'  AS level,
        end_port_id AS key,
        'end_port'  AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY end_port_id
    HAVING COUNT(*) >= 2
),

-- kind: pair_svc
pair_svc_agg AS (
    SELECT
        'pair_svc'                                          AS level,
        CONCAT(port_pair, '_', service_type)               AS key,
        'pair_svc'                                          AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY port_pair, service_type
    HAVING COUNT(*) >= 2
),

-- kind: start_port_svc
sp_svc_agg AS (
    SELECT
        'start_port_svc'                                    AS level,
        CONCAT(start_port_id, '_', service_type)           AS key,
        'start_port_svc'                                    AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY start_port_id, service_type
    HAVING COUNT(*) >= 2
),

-- kind: end_port_svc
ep_svc_agg AS (
    SELECT
        'end_port_svc'                                      AS level,
        CONCAT(end_port_id, '_', service_type)             AS key,
        'end_port_svc'                                      AS kind,
        AVG(target)                                                                     AS pair_mean,
        STDDEV_SAMP(target)                                                             AS pair_std,
        COUNT(*)                                                                        AS pair_count,
        MEDIAN(target)                                                                  AS pair_median,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY target)
            - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY target)                     AS pair_iqr
    FROM filtered
    GROUP BY end_port_id, service_type
    HAVING COUNT(*) >= 2
)

SELECT a.level, a.key, a.kind,
       a.pair_mean, a.pair_std, a.pair_count, a.pair_iqr, a.pair_median,
       gs.global_mean, gs.global_std
FROM (
    SELECT * FROM pair_agg
    UNION ALL SELECT * FROM sp_agg
    UNION ALL SELECT * FROM ep_agg
    UNION ALL SELECT * FROM pair_svc_agg
    UNION ALL SELECT * FROM sp_svc_agg
    UNION ALL SELECT * FROM ep_svc_agg
) a
CROSS JOIN global_stats gs
ORDER BY a.kind, a.key
