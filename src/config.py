"""In-app config. Set by environment variables."""

import os
from typing import Set, Optional
import re
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SBERT_MODELS: list[str] = [
    "BAAI/bge-small-en-v1.5",  # 384 dim
    "BAAI/bge-base-en-v1.5",  # 768 dim
    "msmarco-distilbert-base-tas-b",  # 768 dim
]

LOCAL_DEVELOPMENT: bool = os.getenv("LOCAL_DEVELOPMENT", "False").lower() == "true"
INDEX_ENCODER_CACHE_FOLDER: Optional[str] = (
    os.getenv("INDEX_ENCODER_CACHE_FOLDER", "/models")
    if not LOCAL_DEVELOPMENT
    else None
)
ENCODING_BATCH_SIZE: int = int(os.getenv("ENCODING_BATCH_SIZE", "32"))
# comma-separated 2-letter ISO codes
TARGET_LANGUAGES: Set[str] = set(os.getenv("TARGET_LANGUAGES", "en").lower().split(","))
ENCODER_SUPPORTED_LANGUAGES: Set[str] = {"en"}
FILES_TO_PROCESS = os.getenv("FILES_TO_PROCESS")
BLOCKS_TO_FILTER = os.getenv("BLOCKS_TO_FILTER", "Table,Figure").split(",")
# This matches the ID pattern enforced by the backend, maybe we should share this code?
_ID_ELEMENT = r"[a-zA-Z0-9]+([-_]?[a-zA-Z0-9]+)*"
ID_PATTERN = rf"{_ID_ELEMENT}\.{_ID_ELEMENT}\.{_ID_ELEMENT}\.{_ID_ELEMENT}"
S3_PATTERN = re.compile(r"s3://(?P<bucket>[\w-]+)/(?P<prefix>.+)")
