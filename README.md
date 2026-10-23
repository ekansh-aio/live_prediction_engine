# live_predict

Portable live voyage prediction bundle.

## What it does

```bash
python predict.py --voyage_id 12345
```

It will:

1. fetch one voyage from Snowflake
2. fetch live AIS + STATUSLOG for the current leg
3. build fresh features for Model A, Dynamic, and FCL
4. assemble a voyage remaining-time prediction
5. optionally store the run in MySQL

## Standalone contract

This folder does not import from the rest of the repo.

To run on another machine, copy this folder and make sure the required assets
exist under `live_predict/assets/`.

## Run

```bash
pip install -r requirements.txt
copy .env.example .env
python sf_daemon.py
python predict.py --voyage_id 12345
```

Or set `SF_MODE=direct` to connect without the daemon.

## Required assets

Place these files under `assets/`:

### Required

- `assets/models/model_a.txt`
- `assets/models/dynamic_model_early.txt`
- `assets/models/dynamic_model_late.txt`
- `assets/models/dynamic_feature_lists.json`
- `assets/models/future_leg_correction_model.txt`
- `assets/models/future_leg_correction_lists.json`
- `assets/model_a/route_patterns_v3.csv`
- `assets/model_a/voyage_ais_atl_aggregates_clean.csv`

### Optional

- `assets/lookups/port_lookup.parquet` — fetched live from Snowflake if absent
- `assets/lookups/vessel_features.parquet` — fetched live from Snowflake if absent
- `assets/model_a/subservice_mmsi_map.csv`
- `assets/model_a/voyage_carrier_map.csv`
- `assets/model_a/voyage_extras.csv`
- `assets/features/leg_context_features.parquet`
- `assets/features/bias_correction.json`
- `assets/features/route_marker_wait.csv`
- `assets/features/route_stoppage_patterns.csv`

## MySQL schema

Use `schemas/mysql.sql`.

## Notes

- Model A serving uses `model_a.txt` only.
- Dynamic/FCL assets are required for full live parity.
- Missing optional bias/markerwait/stoppage assets degrade to zero add-ons.
- Port and vessel lookups are fetched from Snowflake when the local parquet
  copies are absent (`sql/port_lookup.sql`, `sql/vessel_features.sql`).
