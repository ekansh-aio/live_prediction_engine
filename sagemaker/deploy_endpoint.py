"""Create or update the SageMaker real-time inference endpoint using boto3 only.
No sagemaker SDK required.

Usage:
    # First deploy
    python sagemaker/deploy_endpoint.py --model-uri s3://opentug-eta-ml-artifacts/live-predict/model-YYYYMMDD-HHMMSS.tar.gz

    # Update existing endpoint (blue/green, no downtime)
    python sagemaker/deploy_endpoint.py --model-uri s3://... --update
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import boto3

ENDPOINT_NAME = "opentug-live-predict-eta"
INSTANCE_TYPE = "ml.m5.xlarge"

# SageMaker-managed SKLearn 1.2-1 container image (us-east-1)
SKLEARN_IMAGE_URI = "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"


def get_execution_role(sm) -> str:
    """Reuse the IAM role from the existing opentug-eta-wave1-endpoint."""
    try:
        ep = sm.describe_endpoint(EndpointName="opentug-eta-wave1-endpoint")
        cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
        model_name = cfg["ProductionVariants"][0]["ModelName"]
        model_desc = sm.describe_model(ModelName=model_name)
        return model_desc["ExecutionRoleArn"]
    except Exception as exc:
        raise RuntimeError(
            f"Could not auto-detect IAM role from existing endpoint: {exc}\n"
            "Pass --role arn:aws:iam::ACCOUNT:role/ROLE_NAME explicitly."
        ) from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="S3 URI of model.tar.gz")
    parser.add_argument("--role", default=None, help="IAM execution role ARN (auto-detected if omitted)")
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--update", action="store_true", help="Update existing endpoint (blue/green)")
    parser.add_argument("--profile", default="opentug")
    args = parser.parse_args()

    sm = boto3.Session(profile_name=args.profile, region_name="us-east-1").client("sagemaker")

    role = args.role or get_execution_role(sm)
    print(f"Using IAM role: {role}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"opentug-live-predict-{timestamp}"
    config_name = f"opentug-live-predict-config-{timestamp}"

    # 1. Create Model
    print(f"Creating SageMaker Model '{model_name}'...")
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": SKLEARN_IMAGE_URI,
            "ModelDataUrl": args.model_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SF_MODE": "none",
            },
        },
    )
    print(f"  Model created.")

    # 2. Create EndpointConfig
    print(f"Creating EndpointConfig '{config_name}'...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": args.instance_type,
        }],
    )
    print(f"  EndpointConfig created.")

    # 3. Create or Update Endpoint
    if args.update:
        print(f"Updating endpoint '{args.endpoint_name}' (blue/green)...")
        sm.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=config_name,
        )
    else:
        print(f"Creating endpoint '{args.endpoint_name}'...")
        sm.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=config_name,
        )

    # 4. Wait until InService
    print("Waiting for endpoint to be InService (this takes ~10 min)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=args.endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 60},
    )
    print(f"\nEndpoint '{args.endpoint_name}' is InService.")
    print("\nTest with:")
    print(f"  import boto3, json")
    print(f"  rt = boto3.Session(profile_name='opentug').client('sagemaker-runtime', region_name='us-east-1')")
    print(f"  resp = rt.invoke_endpoint(EndpointName='{args.endpoint_name}',")
    print(f"      ContentType='application/json', Body=json.dumps(payload, default=str))")
    print(f"  result = json.loads(resp['Body'].read())")


if __name__ == "__main__":
    main()
