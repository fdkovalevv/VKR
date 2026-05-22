from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ozon_counterfeit.data import build_external_test_manifest, load_external_test_frame, merge_manifest_with_frame
from ozon_counterfeit.encoders import FrozenImageEncoder, FrozenTextEncoder
from ozon_counterfeit.features import build_modality_flags, build_text_corpus
from ozon_counterfeit.models import EarlyFusionClassifier
from ozon_counterfeit.training import load_pickle
from ozon_counterfeit.utils import ensure_dir, load_json, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on external ml_ozon_сounterfeit_test.csv.")
    parser.add_argument("--experiment-dir", required=True, help="Directory produced by train_experiment.py")
    parser.add_argument(
        "--test-csv",
        default="data/ml_ozon_сounterfeit_data/ml_ozon_сounterfeit_test.csv",
    )
    parser.add_argument(
        "--test-images-dir",
        default="data/ml_ozon_сounterfeit_test_images",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/predictions/external_test_predictions.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def predict_fusion(
    metadata: dict[str, object],
    experiment_dir: Path,
    test_frame: pd.DataFrame,
    preprocessor,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model = EarlyFusionClassifier(
        tabular_dim=int(metadata["tabular_feature_dim"]),
        text_dim=int(metadata["text_embedding_dim"]),
        image_dim=int(metadata["image_embedding_dim"]),
        projection_dim=int(metadata["projection_dim"]),
        hidden_dim=int(metadata["hidden_dim"]),
        dropout=float(metadata["dropout"]),
    )
    state_dict = torch.load(experiment_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    tabular = preprocessor.transform(test_frame)
    text_encoder = FrozenTextEncoder(str(metadata["text_model_name"]), device=device)
    text = text_encoder.encode(build_text_corpus(test_frame), batch_size=batch_size)

    if metadata["experiment"] == "e2":
        image_encoder = FrozenImageEncoder(str(metadata["image_model_name"]), device=device)
        image = np.zeros((len(test_frame), int(metadata["image_embedding_dim"])), dtype=np.float32)
        has_image_mask = test_frame["has_image"].to_numpy(dtype=bool)
        if has_image_mask.any():
            encoded = image_encoder.encode(
                test_frame.loc[has_image_mask, "image_path"].astype(str).tolist(),
                batch_size=max(8, batch_size // 4),
                num_workers=num_workers,
            )
            image[has_image_mask] = encoded
        flags = build_modality_flags(len(test_frame), test_frame["has_image"].to_numpy(dtype=np.float32))
    else:
        image = np.zeros((len(test_frame), int(metadata["image_embedding_dim"])), dtype=np.float32)
        flags = build_modality_flags(len(test_frame), np.zeros(len(test_frame), dtype=np.float32))

    with torch.no_grad():
        logits = model(
            torch.from_numpy(tabular).to(device),
            torch.from_numpy(text).to(device),
            torch.from_numpy(image).to(device),
            torch.from_numpy(flags).to(device),
        )
        return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    args = parse_args()
    experiment_dir = Path(args.experiment_dir)
    metadata = load_json(experiment_dir / "metadata.json")
    metrics = load_json(experiment_dir / "metrics.json")
    threshold = float(metrics["validation"]["threshold"])
    device = resolve_torch_device(args.device)

    test_source = load_external_test_frame(args.test_csv)
    manifest = build_external_test_manifest(test_source, args.test_images_dir)
    test_frame = merge_manifest_with_frame(test_source, manifest)

    preprocessor = load_pickle(experiment_dir / "tabular_preprocessor.pkl")
    if metadata["experiment"] == "e0":
        model = load_pickle(experiment_dir / "model.pkl")
        scores = model.predict_proba(preprocessor.transform(test_frame))[:, 1]
    else:
        scores = predict_fusion(
            metadata,
            experiment_dir,
            test_frame,
            preprocessor,
            device,
            args.batch_size,
            args.num_workers,
        )

    predictions = (scores >= threshold).astype(int)
    output_frame = pd.DataFrame(
        {
            "id": test_frame["id"],
            "ItemID": test_frame["ItemID"],
            "score": scores,
            "prediction": predictions,
            "has_image": test_frame["has_image"],
        }
    )
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    output_frame.to_csv(output_path, index=False)
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
