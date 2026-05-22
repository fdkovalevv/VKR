from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import ITEM_ID_COLUMN, ROW_ID_COLUMN, SELLER_ID_COLUMN, TARGET_COLUMN


def load_train_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    frame.insert(0, ROW_ID_COLUMN, range(len(frame)))
    frame[ITEM_ID_COLUMN] = frame[ITEM_ID_COLUMN].astype(str)
    frame[SELLER_ID_COLUMN] = frame[SELLER_ID_COLUMN].astype(str)
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    return frame


def load_external_test_frame(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, low_memory=False)
    frame.insert(0, ROW_ID_COLUMN, range(len(frame)))
    frame[ITEM_ID_COLUMN] = frame[ITEM_ID_COLUMN].astype(str)
    frame[SELLER_ID_COLUMN] = frame[SELLER_ID_COLUMN].astype(str)
    return frame


def build_image_mapping(image_dir: str | Path) -> dict[str, str]:
    image_dir = Path(image_dir)
    return {path.stem: str(path.resolve()) for path in image_dir.glob("*.png")}


def build_train_manifest(
    train_frame: pd.DataFrame,
    image_dir: str | Path,
    split_map: dict[str, str],
) -> pd.DataFrame:
    image_map = build_image_mapping(image_dir)
    manifest = train_frame[[ROW_ID_COLUMN, ITEM_ID_COLUMN, SELLER_ID_COLUMN, TARGET_COLUMN]].copy()
    manifest["image_path"] = manifest[ITEM_ID_COLUMN].map(image_map)
    manifest["has_image"] = manifest["image_path"].notna().astype(int)
    manifest["split"] = manifest[SELLER_ID_COLUMN].map(split_map)

    if manifest["split"].isna().any():
        missing_sellers = manifest.loc[manifest["split"].isna(), SELLER_ID_COLUMN].unique().tolist()
        raise ValueError(f"Split assignment is missing for sellers: {missing_sellers[:10]}")

    return manifest


def build_external_test_manifest(
    test_frame: pd.DataFrame,
    image_dir: str | Path,
) -> pd.DataFrame:
    image_map = build_image_mapping(image_dir)
    manifest = test_frame[[ROW_ID_COLUMN, "id", ITEM_ID_COLUMN, SELLER_ID_COLUMN]].copy()
    manifest["image_path"] = manifest[ITEM_ID_COLUMN].map(image_map)
    manifest["has_image"] = manifest["image_path"].notna().astype(int)
    return manifest


def merge_manifest_with_frame(frame: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(manifest, on=ROW_ID_COLUMN, how="inner", suffixes=("", "_manifest"))
