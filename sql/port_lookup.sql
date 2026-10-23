-- port_lookup.sql
-- Port reference data, used when assets/lookups/port_lookup.parquet is absent.
-- Deduplicates to one port area per port (picks the first alphabetically).

SELECT ID, NAME, CITY, STATE, COUNTRY, LON, LAT, PORT_AREA
FROM (
    SELECT
        P.ID,
        P.NAME,
        P.CITY,
        P.STATE,
        P.COUNTRY,
        ST_X(TRY_TO_GEOGRAPHY(P.LOCATION)) AS LON,
        ST_Y(TRY_TO_GEOGRAPHY(P.LOCATION)) AS LAT,
        GPA.NAME AS PORT_AREA,
        ROW_NUMBER() OVER (PARTITION BY P.ID ORDER BY GPA.NAME) AS rn
    FROM EDPPROD.EDP_SOURCE.WEBATDS_PORT P
    LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_PORT_PORT_AREAS PPA ON P.ID = PPA.PORT_ID
    LEFT JOIN EDPPROD.EDP_SOURCE.WEBATDS_GREATERPORTAREA GPA ON PPA.GREATERPORTAREA_ID = GPA.ID
)
WHERE rn = 1
ORDER BY ID
;
