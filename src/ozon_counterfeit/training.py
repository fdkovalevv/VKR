from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import EarlyFusionClassifier
from .utils import ensure_dir, save_json, set_seed


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for display only.
    tqdm = None


def choose_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]
    f1_values = (2 * precision * recall) / np.clip(precision + recall, a_min=1e-8, a_max=None)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_pickle(path: str | Path, payload: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path,
    random_state: int = 42,
) -> dict[str, object]:
    model = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train.astype(int))

    val_scores = model.predict_proba(X_val)[:, 1]
    threshold = choose_threshold_by_f1(y_val, val_scores)
    test_scores = model.predict_proba(X_test)[:, 1]

    metrics = {
        "validation": compute_metrics(y_val, val_scores, threshold),
        "test": compute_metrics(y_test, test_scores, threshold),
    }
    importances = sorted(
        zip(feature_names, model.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    output_dir = ensure_dir(output_dir)
    save_pickle(output_dir / "model.pkl", model)
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "feature_importances.json",
        {"feature_importances": [{"feature": name, "importance": float(score)} for name, score in importances[:100]]},
    )

    return {
        "model": model,
        "threshold": threshold,
        "metrics": metrics,
    }


class FusionArrayDataset(Dataset):
    def __init__(
        self,
        tabular: np.ndarray,
        text: np.ndarray,
        image: np.ndarray,
        modality_flags: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.tabular = torch.from_numpy(tabular.astype(np.float32))
        self.text = torch.from_numpy(text.astype(np.float32))
        self.image = torch.from_numpy(image.astype(np.float32))
        self.modality_flags = torch.from_numpy(modality_flags.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.tabular[index],
            self.text[index],
            self.image[index],
            self.modality_flags[index],
            self.labels[index],
        )


@dataclass
class NeuralTrainingConfig:
    epochs: int = 12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    projection_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.2
    random_state: int = 42
    num_workers: int = 2


def _make_loader(
    tabular: np.ndarray,
    text: np.ndarray,
    image: np.ndarray,
    flags: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FusionArrayDataset(tabular, text, image, flags, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def _predict_scores(model: nn.Module, loader: DataLoader, device: str, desc: str = "Predicting") -> np.ndarray:
    scores = []
    model.eval()
    batch_iterator = loader
    if tqdm is not None:
        batch_iterator = tqdm(loader, desc=desc, leave=False)

    with torch.no_grad():
        for tabular, text, image, flags, _ in batch_iterator:
            logits = model(
                tabular.to(device),
                text.to(device),
                image.to(device),
                flags.to(device),
            )
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores, axis=0)


def train_early_fusion(
    train_inputs: dict[str, np.ndarray],
    val_inputs: dict[str, np.ndarray],
    test_inputs: dict[str, np.ndarray],
    output_dir: str | Path,
    device: str,
    config: NeuralTrainingConfig,
) -> dict[str, object]:
    set_seed(config.random_state)
    output_dir = ensure_dir(output_dir)

    train_loader = _make_loader(
        train_inputs["tabular"],
        train_inputs["text"],
        train_inputs["image"],
        train_inputs["flags"],
        train_inputs["labels"],
        config.batch_size,
        config.num_workers,
        shuffle=True,
    )
    val_loader = _make_loader(
        val_inputs["tabular"],
        val_inputs["text"],
        val_inputs["image"],
        val_inputs["flags"],
        val_inputs["labels"],
        config.batch_size,
        config.num_workers,
        shuffle=False,
    )
    test_loader = _make_loader(
        test_inputs["tabular"],
        test_inputs["text"],
        test_inputs["image"],
        test_inputs["flags"],
        test_inputs["labels"],
        config.batch_size,
        config.num_workers,
        shuffle=False,
    )

    model = EarlyFusionClassifier(
        tabular_dim=train_inputs["tabular"].shape[1],
        text_dim=train_inputs["text"].shape[1],
        image_dim=train_inputs["image"].shape[1],
        projection_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    positive_count = float(train_inputs["labels"].sum())
    negative_count = float(len(train_inputs["labels"]) - positive_count)
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history = []
    best_pr_auc = float("-inf")
    best_state = None
    best_threshold = 0.5
    epochs_without_improvement = 0

    epoch_iterator = range(1, config.epochs + 1)
    if tqdm is not None:
        epoch_iterator = tqdm(epoch_iterator, desc="Training epochs")

    for epoch in epoch_iterator:
        model.train()
        running_loss = 0.0
        train_batch_iterator = train_loader
        if tqdm is not None:
            train_batch_iterator = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{config.epochs}",
                leave=False,
            )

        for tabular, text, image, flags, labels in train_batch_iterator:
            optimizer.zero_grad()
            logits = model(
                tabular.to(device),
                text.to(device),
                image.to(device),
                flags.to(device),
            )
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            if tqdm is not None:
                train_batch_iterator.set_postfix(loss=f"{float(loss.item()):.4f}")

        train_loss = running_loss / max(len(train_inputs["labels"]), 1)
        val_scores = _predict_scores(model, val_loader, device, desc=f"Validation {epoch}/{config.epochs}")
        val_pr_auc = float(average_precision_score(val_inputs["labels"], val_scores))
        val_threshold = choose_threshold_by_f1(val_inputs["labels"], val_scores)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr_auc})

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_threshold = val_threshold
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if tqdm is not None:
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_pr_auc=f"{val_pr_auc:.4f}",
                best_pr_auc=f"{best_pr_auc:.4f}",
            )

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    val_scores = _predict_scores(model, val_loader, device, desc="Final validation")
    test_scores = _predict_scores(model, test_loader, device, desc="Final test")
    metrics = {
        "validation": compute_metrics(val_inputs["labels"], val_scores, best_threshold),
        "test": compute_metrics(test_inputs["labels"], test_scores, best_threshold),
    }

    torch.save(model.state_dict(), output_dir / "model.pt")
    save_json(output_dir / "metrics.json", metrics)

    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_pr_auc"])
        writer.writeheader()
        writer.writerows(history)

    return {
        "model": model,
        "threshold": best_threshold,
        "metrics": metrics,
        "history": history,
    }
