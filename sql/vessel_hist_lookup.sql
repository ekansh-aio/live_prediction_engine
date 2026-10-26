-- Vessel + route historical pace statistics for feature engineering.
-- Computes per-vessel and per-vessel-route delay patterns from all completed legs.
-- Output is joined on MMSI at inference time to populate vessel_hist_* features.

WITH leg_data AS (
    SELECT
        t.MMSI,
        ss.ID                                      AS subservice_id,
        CONCAT(ss.START_PORT_ID, '_', ss.END_PORT_ID) AS port_pair,
        ss.ESTIMATED_DISTANCE / ss.ESTIMATED_SPEED AS planned_h,
        DATEDIFF('second',
            ss.TIME_STARTED,
            ss.TIME_COMPLETED) / 3600.0            AS actual_h
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

vessel_stats AS (
    SELECT
        MMSI,
        COUNT(*)                             AS vessel_hist_leg_count,
        MEDIAN(actual_h / planned_h)         AS vessel_hist_pace_ratio,
        STDDEV(actual_h - planned_h)         AS vessel_hist_delay_std,
        MEDIAN(actual_h / planned_h)         AS vessel_hist_avg_pace
    FROM leg_data
    GROUP BY MMSI
),

route_stats AS (
    SELECT
        MMSI,
        port_pair,
        COUNT(*)                             AS vessel_route_count,
        MEDIAN(actual_h / planned_h)         AS vessel_route_pace_ratio
    FROM leg_data
    GROUP BY MMSI, port_pair
)

SELECT
    v.MMSI,
    v.vessel_hist_leg_count,
    v.vessel_hist_pace_ratio,
    v.vessel_hist_delay_std,
    v.vessel_hist_avg_pace,
    r.port_pair,
    r.vessel_route_count,
    r.vessel_route_pace_ratio
FROM vessel_stats v
LEFT JOIN route_stats r USING (MMSI)
ORDER BY v.MMSI, r.port_pair
