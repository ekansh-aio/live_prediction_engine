from __future__ import annotations

import json
import logging
import os
import sys

log = logging.getLogger(__name__)


def model_fn(model_dir: str) -> dict:
    # SM_MODEL_DIR must be set before any src.* import — config.py reads it at import time
    os.environ["SM_MODEL_DIR"] = model_dir
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from src import markerwait, stoppage
    from src.dynamic_infer import DynamicPredictor
    from src.flc_infer import FLCInfer
    from src.model_a_infer import ModelAInfer

    log.info("Loading models from %s", model_dir)
    preloaded = {
        "model_a": ModelAInfer(client=None),
        "dynamic": DynamicPredictor(),
        "flc": FLCInfer(),
        "marker_map": markerwait.load(),
        "stop_patterns": stoppage.load(),
    }
    log.info("All models loaded")
    return preloaded


def input_fn(request_body: str | bytes, content_type: str) -> dict:
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json.")
    data = json.loads(request_body)
    for key in ("voyage_id", "voyage_legs", "current_leg_ais", "current_leg_statuslog"):
        if key not in data:
            raise ValueError(f"Missing required key in payload: '{key}'")
    return data


def predict_fn(input_data: dict, model: dict) -> dict:
    # Import here (not at module level) to ensure SM_MODEL_DIR is already set
    from src.live_predict import predict_voyage_from_data

    return predict_voyage_from_data(input_data, preloaded=model)


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    return json.dumps(prediction, default=str), "application/json"
