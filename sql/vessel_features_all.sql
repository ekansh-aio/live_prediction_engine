-- vessel_features_all.sql
-- Pre-baked vessel/barge composition for all subservices from the past 3 years.
-- Used to populate assets/lookups/vessel_features.parquet before SageMaker packaging.
--
-- Unlike vessel_features.sql (which filters by specific {leg_ids} at runtime),
-- this fetches all recent subservices so ModelAInfer can look up any voyage
-- without a Snowflake connection inside the SageMaker container.
--
-- Refresh cadence: monthly, or whenever models are re-packaged.
-- Run with:
--   python scripts/refresh_vessel_features.py
--   (or directly via SnowflakeClient.query_file)

SELECT
    sa.RELATED_SAILING_ID                                                          AS subservice_id,

    -- Tow size (barge count)
    COUNT(DISTINCT CASE WHEN b.COMPANYASSET_PTR_ID IS NOT NULL
                        THEN b.COMPANYASSET_PTR_ID END)                            AS tow_size,

    -- Tug specs
    MAX(tug.BHP)                                                                   AS tug_bhp,
    MAX(tug.SPEED_LADEN)                                                           AS tug_speed_laden,
    MAX(tug.SPEED_UPRIVER)                                                         AS tug_speed_upriver,
    MAX(tug.SPEED_DOWNRIVER)                                                       AS tug_speed_downriver,
    MAX(tug.REPORTED_LENGTH)                                                       AS tug_length,
    MAX(tug.REPORTED_BREADTH)                                                      AS tug_breadth,
    MAX(tug.DRAFT)                                                                 AS tug_draft,

    -- Barge composition
    SUM(b.TONNAGE_CAPACITY)                                                        AS total_barge_tonnage,
    AVG(b.DRAFT)                                                                   AS avg_barge_draft,
    SUM(b.DEADWEIGHT_CAPACITY)                                                     AS total_deadweight,

    -- Loaded state
    SUM(sa.LOADED_TONS)                                                            AS total_loaded_tons,
    SUM(sa.LOADED_BARRELS)                                                         AS total_loaded_barrels,

    -- Total asset count (tug + barges)
    COUNT(DISTINCT sa.ASSET_ID)                                                    AS total_assets

FROM EDPPROD.EDP_SOURCE.WEBATDS_SAILINGASSET sa
INNER JOIN EDPPROD.EDP_SOURCE.WEBATDS_SUBSERVICE sub
    ON sub.ID = sa.RELATED_SAILING_ID
LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_TUG tug
    ON tug.COMPANYASSET_PTR_ID = sa.ASSET_ID
LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_BARGE b
    ON b.COMPANYASSET_PTR_ID = sa.ASSET_ID
WHERE sub.IS_DELETED = FALSE
  AND COALESCE(sub.TIME_STARTED, sub.EARLIEST_DEPARTURE) >= DATEADD(year, -3, CURRENT_DATE)
GROUP BY sa.RELATED_SAILING_ID
ORDER BY sa.RELATED_SAILING_ID
;
