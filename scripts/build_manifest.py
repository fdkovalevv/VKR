from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ozon_counterfeit.constants import RANDOM_SEED
from ozon_counterfeit.data import build_train_manifest, load_train_frame
from ozon_counterfeit.splitting import compute_split_summary, greedy_group_split
from ozon_counterfeit.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the train manifest and group-aware split.")
    parser.add_argument(
        "--train-csv",
        default="data/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv",
        help="Path to ml_ozon_сounterfeit_train.csv",
    )
    parser.add_argument(
        "--train-images-dir",
        default="data/ml_ozon_сounterfeit_train_images",
        help="Directory with train images matched by ItemID.",
    )
    parser.add_argument(
        "--output-manifest",
        default="artifacts/manifests/train_manifest.csv",
        help="Where to write the manifest CSV.",
    )
    parser.add_argument(
        "--output-summary",
        default="artifacts/manifests/train_manifest_summary.json",
        help="Where to write split summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_frame = load_train_frame(args.train_csv)
    split_map = greedy_group_split(train_frame, random_state=RANDOM_SEED)
    manifest = build_train_manifest(train_frame, args.train_images_dir, split_map)

    output_manifest = Path(args.output_manifest)
    ensure_dir(output_manifest.parent)
    manifest.to_csv(output_manifest, index=False)

    summary_frame = compute_split_summary(manifest)
    summary_payload = {
        "rows_total": int(len(manifest)),
        "rows_with_image": int(manifest["has_image"].sum()),
        "rows_without_image": int((1 - manifest["has_image"]).sum()),
        "split_summary": summary_frame.to_dict(orient="records"),
    }
    save_json(args.output_summary, summary_payload)

    print(f"Manifest written to {output_manifest}")
    print(summary_frame.to_string(index=False))
    print(
        "Coverage:",
        f"{int(manifest['has_image'].sum())}/{len(manifest)} rows with images",
    )


if __name__ == "__main__":
    main()
