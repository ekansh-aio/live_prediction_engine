"""Build model.tar.gz and upload to S3.

Usage:
    python sagemaker/package_model.py [--bucket BUCKET] [--prefix PREFIX]

Defaults:
    bucket: opentug-eta-model
    prefix: live-predict
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / "assets/lookups/port_lookup.parquet",
    ROOT / "assets/lookups/vessel_features.parquet",
    ROOT / "assets/models/model_a.txt",
    ROOT / "assets/models/dynamic_model_early.txt",
    ROOT / "assets/models/dynamic_model_late.txt",
    ROOT / "assets/models/dynamic_feature_lists.json",
    ROOT / "assets/models/future_leg_correction_model.txt",
    ROOT / "assets/models/future_leg_correction_lists.json",
    ROOT / "assets/model_a/route_patterns_v3.csv",
    ROOT / "assets/model_a/voyage_ais_atl_aggregates_clean.csv",
    ROOT / "assets/features/bias_correction.json",
]

INCLUDE_DIRS = [
    ("sagemaker/inference.py", "code/inference.py"),
    ("sagemaker/requirements.txt", "code/requirements.txt"),
]


def validate():
    missing = [p for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("ERROR: Missing required files before packaging:")
        for p in missing:
            print(f"  - {p.relative_to(ROOT)}")
        sys.exit(1)
    print(f"All {len(REQUIRED_FILES)} required files present.")


def _add_dir(tf: tarfile.TarFile, src_dir: Path, arc_prefix: str) -> int:
    count = 0
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            arc_name = arc_prefix + "/" + f.relative_to(src_dir).as_posix()
            tf.add(str(f), arcname=arc_name)
            count += 1
    return count


def build_tar() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # inference entry point + requirements
        for local_rel, arc_name in INCLUDE_DIRS:
            tf.add(str(ROOT / local_rel), arcname=arc_name)

        # src package
        count = _add_dir(tf, ROOT / "src", "src")
        print(f"  src/: {count} files")

        # assets (models, model_a, features, lookups)
        for subdir in ("models", "model_a", "features", "lookups"):
            src = ROOT / "assets" / subdir
            if src.exists():
                count = _add_dir(tf, src, f"assets/{subdir}")
                print(f"  assets/{subdir}/: {count} files")

    buf.seek(0)
    return buf.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="opentug-eta-ml-artifacts")
    parser.add_argument("--prefix", default="live-predict")
    parser.add_argument("--profile", default="opentug")
    args = parser.parse_args()

    validate()

    print("Building model.tar.gz...")
    data = build_tar()
    size_mb = len(data) / 1024 / 1024
    print(f"Archive size: {size_mb:.1f} MB")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    s3_key = f"{args.prefix}/model-{timestamp}.tar.gz"
    s3_uri = f"s3://{args.bucket}/{s3_key}"

    print(f"Uploading to {s3_uri} ...")
    session = boto3.Session(profile_name=args.profile, region_name="us-east-1")
    s3 = session.client("s3")
    s3.put_object(Bucket=args.bucket, Key=s3_key, Body=data)
    print(f"\nDone. Model URI:\n  {s3_uri}")
    print("\nNext step:")
    print(f"  python sagemaker/deploy_endpoint.py --model-uri {s3_uri}")


if __name__ == "__main__":
    main()
