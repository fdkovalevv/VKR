from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ozon_counterfeit.data import load_train_frame
from ozon_counterfeit.splitting import compute_split_summary
from ozon_counterfeit.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data checks for the train manifest.")
    parser.add_argument(
        "--train-csv",
        default="data/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv",
    )
    parser.add_argument(
        "--manifest-path",
        default="artifacts/manifests/train_manifest.csv",
    )
    parser.add_argument(
        "--output-report",
        default="artifacts/reports/data_checks.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_frame = load_train_frame(args.train_csv)
    manifest = train_frame[["row_id", "ItemID", "SellerID", "resolution"]].merge(
        pd.read_csv(args.manifest_path),
        on=["row_id", "ItemID", "SellerID", "resolution"],
        how="inner",
    )

    if len(manifest) != len(train_frame):
        raise ValueError("Manifest row count does not match the train CSV row count.")

    exact_mapping_ok = bool(
        manifest.loc[manifest["has_image"] == 1, "image_path"]
        .map(lambda value: Path(value).stem)
        .eq(manifest.loc[manifest["has_image"] == 1, "ItemID"])
        .all()
    )

    split_seller_sets = {
        split_name: set(split_frame["SellerID"].astype(str))
        for split_name, split_frame in manifest.groupby("split")
    }
    overlap_report = {}
    split_names = sorted(split_seller_sets)
    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_seller_sets[left_name] & split_seller_sets[right_name]
            overlap_report[f"{left_name}__{right_name}"] = len(overlap)

    summary_frame = compute_split_summary(manifest)
    report = {
        "manifest_rows": int(len(manifest)),
        "train_rows": int(len(train_frame)),
        "rows_with_image": int(manifest["has_image"].sum()),
        "rows_without_image": int((1 - manifest["has_image"]).sum()),
        "exact_mapping_ok": exact_mapping_ok,
        "seller_overlap_counts": overlap_report,
        "split_summary": summary_frame.to_dict(orient="records"),
    }
    save_json(args.output_report, report)

    print(f"Data check report written to {args.output_report}")
    print(summary_frame.to_string(index=False))
    print("Seller overlap:", overlap_report)
    print("Exact ItemID -> image_path mapping:", exact_mapping_ok)


if __name__ == "__main__":
    main()
