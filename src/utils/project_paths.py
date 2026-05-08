import os
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[2]


def default_data_root():
    return Path(os.environ.get("KITTI_SEQUENCES_DIR", project_root() / "data" / "dataset" / "sequences"))


def parse_sequences(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part.strip() for part in value.split(",") if part.strip()]
