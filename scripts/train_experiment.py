from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ozon_counterfeit.constants import (
    DEFAULT_IMAGE_EMBED_DIM,
    DEFAULT_IMAGE_MODEL_NAME,
    DEFAULT_TEXT_MODEL_NAME,
    RANDOM_SEED,
)
from ozon_counterfeit.data import load_train_frame, merge_manifest_with_frame
from ozon_counterfeit.encoders import (
    FrozenImageEncoder,
    FrozenTextEncoder,
    load_cached_array,
    save_cached_array,
)
from ozon_counterfeit.features import (
    TabularPreprocessor,
    build_modality_flags,
    build_text_corpus,
    extract_labels,
    infer_log_columns,
    infer_numeric_columns,
)
from ozon_counterfeit.training import NeuralTrainingConfig, save_pickle, train_early_fusion, train_random_forest
from ozon_counterfeit.utils import ensure_dir, resolve_torch_device, save_json, set_seed, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E0/E1/E2 counterfeit detection experiments.")
    parser.add_argument("--experiment", choices=["e0", "e1", "e2"], required=True)
    parser.add_argument(
        "--train-csv",
        default="data/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_train.csv",
    )
    parser.add_argument(
        "--manifest-path",
        default="artifacts/manifests/train_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Experiment output directory. Defaults to artifacts/experiments/<experiment>.",
    )
    parser.add_argument("--text-model-name", default=DEFAULT_TEXT_MODEL_NAME)
    parser.add_argument("--image-model-name", default=DEFAULT_IMAGE_MODEL_NAME)
    parser.add_argument("--image-embedding-dim", type=int, default=DEFAULT_IMAGE_EMBED_DIM)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--include-missing-images", action="store_true")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    return parser.parse_args()


def sample_per_split(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    sampled = []
    for _, split_frame in frame.groupby("split", sort=False):
        take = min(len(split_frame), limit)
        sampled.append(split_frame.sample(n=take, random_state=RANDOM_SEED))
    return pd.concat(sampled, axis=0).sort_values(["split", "row_id"]).reset_index(drop=True)


def maybe_encode_text(
    split_frame: pd.DataFrame,
    cache_path: Path,
    encoder: FrozenTextEncoder,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    cached = None if force else load_cached_array(cache_path)
    if cached is not None:
        return cached

    texts = build_text_corpus(split_frame)
    embeddings = encoder.encode(texts, batch_size=batch_size)
    return save_cached_array(cache_path, embeddings)


def maybe_encode_images(
    split_frame: pd.DataFrame,
    cache_path: Path,
    encoder: FrozenImageEncoder,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> np.ndarray:
    cached = None if force else load_cached_array(cache_path)
    if cached is not None:
        return cached

    image_paths = split_frame["image_path"].astype(str).tolist()
    embeddings = encoder.encode(
        image_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        desc=f"Encoding images: {cache_path.stem}",
    )
    return save_cached_array(cache_path, embeddings)


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    numeric_columns: list[str],
    log_columns: list[str],
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    image_dim: int,
    text_dim: int,
    tabular_dim: int,
) -> None:
    payload = {
        "experiment": args.experiment,
        "train_csv": args.train_csv,
        "manifest_path": args.manifest_path,
        "text_model_name": args.text_model_name,
        "image_model_name": args.image_model_name,
        "image_embedding_dim": image_dim,
        "text_embedding_dim": text_dim,
        "tabular_feature_dim": tabular_dim,
        "numeric_columns": numeric_columns,
        "log_columns": log_columns,
        "train_rows": int(len(train_frame)),
        "headline_eval_rows": int(len(eval_frame)),
        "include_missing_images": bool(args.include_missing_images),
        "max_samples_per_split": args.max_samples_per_split,
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
    }
    save_json(output_dir / "metadata.json", payload)


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    output_dir = Path(args.output_dir or f"artifacts/experiments/{args.experiment}")
    cache_dir = ensure_dir(output_dir / "cache")
    ensure_dir(output_dir)

    train_frame = load_train_frame(args.train_csv)
    manifest = pd.read_csv(args.manifest_path)
    dataset = merge_manifest_with_frame(train_frame, manifest)

    if args.max_samples_per_split:
        dataset = sample_per_split(dataset, args.max_samples_per_split)

    if args.experiment == "e2" or not args.include_missing_images:
        dataset = dataset.loc[dataset["has_image"] == 1].copy()

    split_frames = {
        split_name: split_frame.reset_index(drop=True)
        for split_name, split_frame in dataset.groupby("split", sort=False)
    }
    required_splits = {"train", "val", "test"}
    missing = required_splits - set(split_frames)
    if missing:
        raise ValueError(f"Missing splits in manifest: {sorted(missing)}")

    numeric_columns = infer_numeric_columns(train_frame)
    log_columns = infer_log_columns(numeric_columns)
    preprocessor = TabularPreprocessor(numeric_columns=numeric_columns, log_columns=log_columns)

    X_train_tab = preprocessor.fit_transform(split_frames["train"])
    X_val_tab = preprocessor.transform(split_frames["val"])
    X_test_tab = preprocessor.transform(split_frames["test"])
    y_train = extract_labels(split_frames["train"])
    y_val = extract_labels(split_frames["val"])
    y_test = extract_labels(split_frames["test"])

    save_pickle(output_dir / "tabular_preprocessor.pkl", preprocessor)

    if args.experiment == "e0":
        results = train_random_forest(
            X_train_tab,
            y_train,
            X_val_tab,
            y_val,
            X_test_tab,
            y_test,
            feature_names=preprocessor.get_feature_names(),
            output_dir=output_dir,
            random_state=RANDOM_SEED,
        )
        save_metadata(
            output_dir,
            args,
            numeric_columns,
            log_columns,
            split_frames["train"],
            split_frames["test"],
            image_dim=0,
            text_dim=0,
            tabular_dim=X_train_tab.shape[1],
        )
        print("E0 test metrics:", results["metrics"]["test"])
        return

    text_encoder = FrozenTextEncoder(args.text_model_name, device=args.device)
    text_train = maybe_encode_text(
        split_frames["train"],
        cache_dir / f"text_train_{slugify(args.text_model_name)}.npy",
        text_encoder,
        batch_size=args.batch_size,
        force=args.force_recompute_embeddings,
    )
    text_val = maybe_encode_text(
        split_frames["val"],
        cache_dir / f"text_val_{slugify(args.text_model_name)}.npy",
        text_encoder,
        batch_size=args.batch_size,
        force=args.force_recompute_embeddings,
    )
    text_test = maybe_encode_text(
        split_frames["test"],
        cache_dir / f"text_test_{slugify(args.text_model_name)}.npy",
        text_encoder,
        batch_size=args.batch_size,
        force=args.force_recompute_embeddings,
    )

    if args.experiment == "e2":
        image_encoder = FrozenImageEncoder(args.image_model_name, device=args.device)
        image_train = maybe_encode_images(
            split_frames["train"],
            cache_dir / f"image_train_{slugify(args.image_model_name)}.npy",
            image_encoder,
            batch_size=max(8, args.batch_size // 4),
            num_workers=args.num_workers,
            force=args.force_recompute_embeddings,
        )
        image_val = maybe_encode_images(
            split_frames["val"],
            cache_dir / f"image_val_{slugify(args.image_model_name)}.npy",
            image_encoder,
            batch_size=max(8, args.batch_size // 4),
            num_workers=args.num_workers,
            force=args.force_recompute_embeddings,
        )
        image_test = maybe_encode_images(
            split_frames["test"],
            cache_dir / f"image_test_{slugify(args.image_model_name)}.npy",
            image_encoder,
            batch_size=max(8, args.batch_size // 4),
            num_workers=args.num_workers,
            force=args.force_recompute_embeddings,
        )
        image_dim = image_train.shape[1]
        train_flags = build_modality_flags(len(split_frames["train"]), np.ones(len(split_frames["train"]), dtype=np.float32))
        val_flags = build_modality_flags(len(split_frames["val"]), np.ones(len(split_frames["val"]), dtype=np.float32))
        test_flags = build_modality_flags(len(split_frames["test"]), np.ones(len(split_frames["test"]), dtype=np.float32))
    else:
        image_dim = args.image_embedding_dim
        image_train = np.zeros((len(split_frames["train"]), image_dim), dtype=np.float32)
        image_val = np.zeros((len(split_frames["val"]), image_dim), dtype=np.float32)
        image_test = np.zeros((len(split_frames["test"]), image_dim), dtype=np.float32)
        train_flags = build_modality_flags(len(split_frames["train"]), np.zeros(len(split_frames["train"]), dtype=np.float32))
        val_flags = build_modality_flags(len(split_frames["val"]), np.zeros(len(split_frames["val"]), dtype=np.float32))
        test_flags = build_modality_flags(len(split_frames["test"]), np.zeros(len(split_frames["test"]), dtype=np.float32))

    device = resolve_torch_device(args.device)
    config = NeuralTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        random_state=RANDOM_SEED,
        num_workers=args.num_workers,
    )

    train_inputs = {
        "tabular": X_train_tab,
        "text": text_train,
        "image": image_train,
        "flags": train_flags,
        "labels": y_train,
    }
    val_inputs = {
        "tabular": X_val_tab,
        "text": text_val,
        "image": image_val,
        "flags": val_flags,
        "labels": y_val,
    }
    test_inputs = {
        "tabular": X_test_tab,
        "text": text_test,
        "image": image_test,
        "flags": test_flags,
        "labels": y_test,
    }

    results = train_early_fusion(
        train_inputs=train_inputs,
        val_inputs=val_inputs,
        test_inputs=test_inputs,
        output_dir=output_dir,
        device=device,
        config=config,
    )

    save_metadata(
        output_dir,
        args,
        numeric_columns,
        log_columns,
        split_frames["train"],
        split_frames["test"],
        image_dim=image_dim,
        text_dim=text_train.shape[1],
        tabular_dim=X_train_tab.shape[1],
    )
    print(f"{args.experiment.upper()} test metrics:", results["metrics"]["test"])


if __name__ == "__main__":
    main()
