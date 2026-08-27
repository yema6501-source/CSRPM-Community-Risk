from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data_pipeline import ChicagoDataConfig, acquire_chicago_dataset
from src.utils import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Download and construct the real-world CSRPM dataset.")
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--output", default="runtime_data")
    parser.add_argument("--app-token", default=os.getenv("CHICAGO_APP_TOKEN"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)["chicago"]
    data_cfg = ChicagoDataConfig(
        start_date=str(cfg["start_date"]),
        end_date=str(cfg["end_date"]),
        frequency=str(cfg["frequency"]),
        page_size=int(cfg["page_size"]),
        timeout=int(cfg["timeout"]),
    )
    result = acquire_chicago_dataset(args.output, data_cfg, app_token=args.app_token)
    panel = result["panel"]
    print(f"Created {args.output}/chicago_panel.csv with {len(panel):,} community-time rows.")
    print(f"Communities: {panel['community_id'].nunique()} | Time points: {panel['date'].nunique()}")


if __name__ == "__main__":
    main()
