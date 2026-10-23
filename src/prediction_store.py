from __future__ import annotations

import json
import time
from pathlib import Path

from .utils import jsonable

STORE_DIR = Path(__file__).resolve().parents[1] / "predictions"


def store_prediction(result: dict) -> str:
    STORE_DIR.mkdir(exist_ok=True)
    voyage_id = result["voyage_id"]
    ts = int(time.time())
    run_id = f"{voyage_id}_{ts}"

    payload: dict = jsonable({**result, "run_id": run_id})

    versioned = STORE_DIR / f"{run_id}.json"
    versioned.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    latest = STORE_DIR / f"voyage_{voyage_id}_latest.json"
    latest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    return run_id
