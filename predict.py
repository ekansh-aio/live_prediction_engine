from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.live_predict import predict_voyage_live  # noqa: E402


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Live voyage ETA predictor")
    ap.add_argument("--voyage_id", type=int, required=True)
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("predict")
    log.info("Starting live prediction for voyage_id=%s store=%s", args.voyage_id, not args.no_store)

    try:
        result = predict_voyage_live(args.voyage_id, store=not args.no_store)
    except Exception:
        log.exception("Prediction failed for voyage_id=%s", args.voyage_id)
        return 1

    log.info("Prediction complete for voyage_id=%s", args.voyage_id)
    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(json.dumps(result, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
