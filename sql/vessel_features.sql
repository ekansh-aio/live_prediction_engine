-- vessel_features.sql
-- Vessel/barge composition per subservice, used when
-- assets/lookups/vessel_features.parquet is absent.
-- {leg_ids} is replaced with the voyage's leg IDs (integers).

SELECT
    sa.RELATED_SAILING_ID AS subservice_id,

    -- Tow size (barge count)
    COUNT(DISTINCT CASE WHEN b.COMPANYASSET_PTR_ID IS NOT NULL THEN b.COMPANYASSET_PTR_ID END) AS tow_size,

    -- Tug specs (take first tug per subservice)
    MAX(tug.BHP) AS tug_bhp,
    MAX(tug.SPEED_LADEN) AS tug_speed_laden,
    MAX(tug.SPEED_UPRIVER) AS tug_speed_upriver,
    MAX(tug.SPEED_DOWNRIVER) AS tug_speed_downriver,
    MAX(tug.REPORTED_LENGTH) AS tug_length,
    MAX(tug.REPORTED_BREADTH) AS tug_breadth,
    MAX(tug.DRAFT) AS tug_draft,

    -- Barge composition
    SUM(b.TONNAGE_CAPACITY) AS total_barge_tonnage,
    AVG(b.DRAFT) AS avg_barge_draft,
    SUM(b.DEADWEIGHT_CAPACITY) AS total_deadweight,

    -- Loaded state
    SUM(sa.LOADED_TONS) AS total_loaded_tons,
    SUM(sa.LOADED_BARRELS) AS total_loaded_barrels,

    -- Number of assets (tug + barge combined)
    COUNT(DISTINCT sa.ASSET_ID) AS total_assets

FROM EDPPROD.EDP_SOURCE.WEBATDS_SAILINGASSET sa
LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_TUG tug ON tug.COMPANYASSET_PTR_ID = sa.ASSET_ID
LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_BARGE b ON b.COMPANYASSET_PTR_ID = sa.ASSET_ID
WHERE sa.RELATED_SAILING_ID IN ({leg_ids})
GROUP BY sa.RELATED_SAILING_ID
;
