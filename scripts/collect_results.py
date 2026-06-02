from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ozon_counterfeit.utils import ensure_dir, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment metrics into one ablation table.")
    parser.add_argument(
        "--experiments-root",
        default="artifacts/experiments",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/reports/ablation_table.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    rows = []
    for experiment_dir in sorted(experiments_root.glob("*")):
        metrics_path = experiment_dir / "metrics.json"
        metadata_path = experiment_dir / "metadata.json"
        if not metrics_path.exists() or not metadata_path.exists():
            continue

        metrics = load_json(metrics_path)
        metadata = load_json(metadata_path)
        rows.append(
            {
                "experiment": metadata["experiment"],
                "pr_auc": metrics["test"]["pr_auc"],
                "precision": metrics["test"]["precision"],
                "recall": metrics["test"]["recall"],
                "f1": metrics["test"]["f1"],
                "threshold": metrics["test"]["threshold"],
                "headline_eval_rows": metadata["headline_eval_rows"],
            }
        )

    if not rows:
        raise FileNotFoundError(f"No completed experiments found under {experiments_root}")

    output_frame = pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    output_frame.to_csv(output_path, index=False)
    print(output_frame.to_string(index=False))
    print(f"Ablation table written to {output_path}")


if __name__ == "__main__":
    main()
