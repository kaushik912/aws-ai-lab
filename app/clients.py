"""AWS clients. No keys anywhere: boto3's default chain (env/SSO profile locally, exec role on Lambda)."""
from functools import lru_cache

import boto3
from botocore.config import Config

from .config import get_settings


@lru_cache
def bedrock():
    s = get_settings()
    return boto3.client(
        "bedrock-runtime",
        region_name=s.aws_region,
        config=Config(retries={"mode": "adaptive", "max_attempts": 5}, read_timeout=60),
    )
