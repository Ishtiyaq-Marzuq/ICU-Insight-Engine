"""
07_logging.py

Simple experiment logger wrapper. For production, hook into MLflow or Weights & Biases.

Provides:
- log_experiment(meta, metrics, artifacts)
"""

import json
from datetime import datetime
import config
import os
from pathlib import Path

def log_experiment(name: str, metrics: dict, params: dict = None, artifacts: list = None):
    timestamp = datetime.utcnow().isoformat()
    record = {
        "name": name,
        "timestamp": timestamp,
        "params": params or {},
        "metrics": metrics or {},
        "artifacts": artifacts or []
    }
    outdir = Path(config.LOG_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    fname = outdir / f"{name.replace(' ','_')}_{timestamp.replace(':','-')}.json"
    with open(fname, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Logged experiment to {fname}")
    return fname
