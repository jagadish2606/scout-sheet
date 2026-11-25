import json
import os
from compression import zstd
from pathlib import Path


DATA_DIR = Path(os.environ.get("STL_DATA_DIR", Path(__file__).parent / "../stl-hackathon-2026-grid-starter"))


def read_jsonl_zst(path):
    with zstd.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            yield json.loads(line)