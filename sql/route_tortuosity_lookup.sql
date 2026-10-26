-- Per route (port_pair) median planned distance.
-- Used to compute tortuosity = planned_dist / haversine(start, end).
-- Tortuosity corrects the GPS haversine undercount for non-straight routes
-- (river channels, coastal routes, canal transits).
-- Minimum 2 observations required for reliability.

SELECT
    ss.START_PORT_ID,
    ss.END_PORT_ID,
    CONCAT(ss.START_PORT_ID, '_', ss.END_PORT_ID) AS port_pair,
    MEDIAN(ss.ESTIMATED_DISTANCE)                  AS route_planned_dist_nm,
    COUNT(*)                                        AS route_obs_count
FROM EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE ss
WHERE ss.IS_DELETED      = FALSE
  AND ss.IS_CURRENT      = TRUE
  AND ss.ESTIMATED_DISTANCE > 0
  AND ss.ESTIMATED_SPEED    > 0
GROUP BY ss.START_PORT_ID, ss.END_PORT_ID
HAVING COUNT(*) >= 2
ORDER BY ss.START_PORT_ID, ss.END_PORT_ID
