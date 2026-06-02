from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "artifacts" / "cache"
MPLCONFIGDIR = CACHE_DIR / "matplotlib"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


@dataclass(frozen=True)
class ComparisonSet:
    key: str
    title: str
    experiments: tuple[str, ...]


COMPARISON_SETS = (
    ComparisonSet(
        key="unimodal",
        title="Unimodal models",
        experiments=("tabular_only", "text_only", "image_only"),
    ),
    ComparisonSet(
        key="best_bimodal",
        title="Best bimodal configurations",
        experiments=("tabular_text_early_fusion", "text_image_simple", "tabular_image_simple"),
    ),
    ComparisonSet(
        key="best_vs_final",
        title="Best stages vs final model",
        experiments=(
            "text_only",
            "tabular_text_early_fusion",
            "tabular_text_image_early_fusion",
            "tabular_text_image_gated_fusion",
            "late_best_non_image_plus_best_image_side",
        ),
    ),
)

DISPLAY_NAMES = {
    "tabular_only": "Tabular only",
    "text_only": "Text only",
    "image_only": "Image only",
    "tabular_text_early_fusion": "Tabular + text",
    "text_image_simple": "Text + image",
    "tabular_image_simple": "Tabular + image",
    "tabular_text_image_early_fusion": "All modalities early",
    "tabular_text_image_gated_fusion": "All modalities gated",
    "late_best_non_image_plus_best_image_side": "Final late fusion",
}

RUSSIAN_DISPLAY_NAMES = {
    "text_only": "Только текст",
    "tabular_text_early_fusion": "Табличные данные + текст",
    "tabular_text_image_early_fusion": "Все модальности, раннее объединение",
    "tabular_text_image_gated_fusion": "Все модальности, управляемое объединение",
    "late_best_non_image_plus_best_image_side": "Финальная модель позднего объединения",
}

COLORS = {
    "tabular_only": "#4C78A8",
    "text_only": "#F58518",
    "image_only": "#54A24B",
    "tabular_text_early_fusion": "#B279A2",
    "text_image_simple": "#E45756",
    "tabular_image_simple": "#72B7B2",
    "tabular_text_image_early_fusion": "#9D755D",
    "tabular_text_image_gated_fusion": "#EECA3B",
    "late_best_non_image_plus_best_image_side": "#000000",
}

SPLIT_KEYS = {
    "validation": ("y_val", "val_scores", "val_pr_auc", "validation"),
    "val": ("y_val", "val_scores", "val_pr_auc", "validation"),
    "test": ("y_test", "test_scores", "test_pr_auc", "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PR-AUC curves for selected multimodal counterfeit experiments."
    )
    parser.add_argument(
        "--experiments-root",
        default=str(PROJECT_ROOT / "artifacts/experiments/all_modalities_efficientnet_b0"),
        help="Directory containing experiment subdirectories with scores_val_test.npz.",
    )
    parser.add_argument(
        "--results-csv",
        default=str(PROJECT_ROOT / "artifacts/reports/all_modalities_efficientnet_b0_results.csv"),
        help="CSV with expected val_pr_auc and test_pr_auc values.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts/reports/figures"),
        help="Directory where PNG figures and the selected-configuration CSV will be saved.",
    )
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated split list. Supported values: validation, val, test.",
    )
    return parser.parse_args()


def parse_splits(raw_splits: str) -> list[str]:
    splits = []
    for raw_split in raw_splits.split(","):
        split = raw_split.strip().lower()
        if not split:
            continue
        if split not in SPLIT_KEYS:
            supported = ", ".join(sorted(SPLIT_KEYS))
            raise ValueError(f"Unsupported split '{split}'. Supported values: {supported}.")
        canonical_split = SPLIT_KEYS[split][3]
        if canonical_split not in splits:
            splits.append(canonical_split)

    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def load_expected_metrics(results_csv: Path) -> dict[str, dict[str, float]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    expected: dict[str, dict[str, float]] = {}
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"experiment", "val_pr_auc", "test_pr_auc"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Results CSV is missing required columns: {missing}")

        for row in reader:
            expected[row["experiment"]] = {
                "val_pr_auc": float(row["val_pr_auc"]),
                "test_pr_auc": float(row["test_pr_auc"]),
            }

    return expected


def load_split_scores(experiment_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    y_key, scores_key, _, _ = SPLIT_KEYS[split]
    scores_path = experiment_dir / "scores_val_test.npz"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    with np.load(scores_path) as payload:
        missing_keys = {y_key, scores_key}.difference(payload.files)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"{scores_path} is missing required arrays: {missing}")
        y_true = np.asarray(payload[y_key], dtype=np.int8)
        y_score = np.asarray(payload[scores_key], dtype=np.float64)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError(f"{scores_path} must contain one-dimensional {y_key} and {scores_key} arrays.")
    if len(y_true) != len(y_score):
        raise ValueError(
            f"{scores_path} has mismatched lengths: {y_key}={len(y_true)}, {scores_key}={len(y_score)}."
        )
    if len(y_true) == 0:
        raise ValueError(f"{scores_path} contains no rows for split '{split}'.")
    if not np.isfinite(y_score).all():
        raise ValueError(f"{scores_path} contains non-finite scores in {scores_key}.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(f"{scores_path} contains labels outside {{0, 1}} in {y_key}.")

    return y_true, y_score


def validate_pr_auc(
    experiment: str,
    split: str,
    pr_auc: float,
    expected_metrics: dict[str, dict[str, float]],
    tolerance: float = 1e-6,
) -> tuple[float, float]:
    _, _, metric_key, _ = SPLIT_KEYS[split]
    if experiment not in expected_metrics:
        raise ValueError(f"Experiment '{experiment}' is missing from the results CSV.")

    expected_pr_auc = expected_metrics[experiment][metric_key]
    absolute_difference = abs(pr_auc - expected_pr_auc)
    if absolute_difference > tolerance:
        raise ValueError(
            f"PR-AUC mismatch for {experiment} on {split}: "
            f"computed={pr_auc:.12f}, expected={expected_pr_auc:.12f}, diff={absolute_difference:.12g}."
        )
    return expected_pr_auc, absolute_difference


def collect_curve_data(
    experiments_root: Path,
    expected_metrics: dict[str, dict[str, float]],
    split: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    curve_data: dict[str, dict[str, object]] = {}
    csv_rows: list[dict[str, object]] = []
    reference_labels: np.ndarray | None = None

    selected_experiments = tuple(dict.fromkeys(experiment for group in COMPARISON_SETS for experiment in group.experiments))
    for experiment in selected_experiments:
        experiment_dir = experiments_root / experiment
        y_true, y_score = load_split_scores(experiment_dir, split)
        if reference_labels is None:
            reference_labels = y_true
        elif not np.array_equal(reference_labels, y_true):
            raise ValueError(f"Label arrays are not aligned for experiment '{experiment}' on split '{split}'.")

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(average_precision_score(y_true, y_score))
        expected_pr_auc, absolute_difference = validate_pr_auc(experiment, split, pr_auc, expected_metrics)
        positive_rate = float(y_true.mean())

        curve_data[experiment] = {
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
            "positive_rate": positive_rate,
        }

        for comparison in COMPARISON_SETS:
            if experiment not in comparison.experiments:
                continue
            csv_rows.append(
                {
                    "split": split,
                    "comparison": comparison.key,
                    "experiment": experiment,
                    "display_name": DISPLAY_NAMES.get(experiment, experiment),
                    "pr_auc": f"{pr_auc:.12f}",
                    "expected_pr_auc": f"{expected_pr_auc:.12f}",
                    "absolute_difference": f"{absolute_difference:.12g}",
                    "positive_rate": f"{positive_rate:.12f}",
                    "score_path": str(experiments_root / experiment / "scores_val_test.npz"),
                }
            )

    return curve_data, csv_rows


def plot_comparison(
    ax: plt.Axes,
    curve_data: dict[str, dict[str, object]],
    comparison: ComparisonSet,
    *,
    title: str | None = None,
    display_names: dict[str, str] | None = None,
    positive_rate_label: str = "Positive rate",
    legend_fontsize: float = 8.2,
) -> None:
    display_names = DISPLAY_NAMES if display_names is None else display_names
    first_experiment = comparison.experiments[0]
    positive_rate = float(curve_data[first_experiment]["positive_rate"])
    ax.axhline(
        positive_rate,
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        label=f"{positive_rate_label}={positive_rate:.3f}",
    )

    for experiment in comparison.experiments:
        data = curve_data[experiment]
        line_width = 2.6 if experiment == "late_best_non_image_plus_best_image_side" else 2.0
        ax.plot(
            data["recall"],
            data["precision"],
            color=COLORS.get(experiment),
            linewidth=line_width,
            label=f"{display_names.get(experiment, experiment)} (PR-AUC={data['pr_auc']:.3f})",
        )

    ax.set_title(comparison.title if title is None else title, fontsize=12, pad=7)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.7)
    ax.legend(loc="lower left", fontsize=legend_fontsize, frameon=True)


def plot_split(curve_data: dict[str, dict[str, object]], split: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(COMPARISON_SETS), figsize=(18, 5.4), sharex=True, sharey=True)
    if len(COMPARISON_SETS) == 1:
        axes = [axes]

    split_title = "validation" if split == "validation" else "test"
    for ax, comparison in zip(axes, COMPARISON_SETS, strict=True):
        plot_comparison(ax, curve_data, comparison)

    axes[0].set_ylabel("Precision")
    for ax in axes:
        ax.set_xlabel("Recall")

    fig.suptitle(f"Precision-Recall curves on {split_title} split", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_single_comparison(
    curve_data: dict[str, dict[str, object]],
    split: str,
    comparison: ComparisonSet,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    plot_comparison(
        ax,
        curve_data,
        comparison,
        title="Лучшие этапы и финальная модель",
        display_names=RUSSIAN_DISPLAY_NAMES,
        positive_rate_label="Доля положительного класса",
        legend_fontsize=8.0,
    )

    split_title = "валидационной" if split == "validation" else "тестовой"
    ax.set_xlabel("Полнота")
    ax.set_ylabel("Точность")

    fig.suptitle(f"PR-кривые на {split_title} выборке", fontsize=14, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.86)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_selected_configurations(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "comparison",
        "experiment",
        "display_name",
        "pr_auc",
        "expected_pr_auc",
        "absolute_difference",
        "positive_rate",
        "score_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiments_root = Path(args.experiments_root)
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = parse_splits(args.splits)
    expected_metrics = load_expected_metrics(results_csv)

    all_rows: list[dict[str, object]] = []
    output_paths = []
    best_vs_final = next(comparison for comparison in COMPARISON_SETS if comparison.key == "best_vs_final")
    for split in splits:
        curve_data, csv_rows = collect_curve_data(experiments_root, expected_metrics, split)
        output_path = output_dir / f"pr_auc_comparisons_{split}.png"
        plot_split(curve_data, split, output_path)
        best_vs_final_output_path = output_dir / f"pr_auc_best_vs_final_{split}.png"
        plot_single_comparison(curve_data, split, best_vs_final, best_vs_final_output_path)
        all_rows.extend(csv_rows)
        output_paths.append(output_path)
        output_paths.append(best_vs_final_output_path)

    selected_csv = output_dir / "pr_auc_selected_configurations.csv"
    write_selected_configurations(all_rows, selected_csv)

    print("Saved PR-AUC comparison figures:")
    for output_path in output_paths:
        print(f"- {output_path}")
    print(f"Saved selected-configuration metrics: {selected_csv}")


if __name__ == "__main__":
    main()
