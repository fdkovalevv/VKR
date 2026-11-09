# Multimodal Counterfeit Product Detection on Ozon

[Русская версия](README.ru.md)

This repository contains the code for a bachelor’s thesis on binary classification of Ozon product listings. The task is to determine whether a product is classified as counterfeit or otherwise problematic according to the target label `resolution`.

The thesis compares models that use different sources of data:

- tabular product and seller features
- text fields from product listings
- product images
- multimodal combinations of these features

## Project Structure

- `src/ozon_counterfeit/` - core code for loading data, preparing features, splitting data, defining models, and training.
- `scripts/` - scripts for building the manifest, checking data, training models, running inference, and generating plots.
- `*.ipynb` - notebooks containing experiments and analysis.
- `docs/` - a brief description of the project pipeline.
- `artifacts/` - run outputs, including the manifest, metrics, plots, predictions, and intermediate files.

## Data

Expected data directory structure:

```text
data/
  ml_ozon_сounterfeit_data/
    ml_ozon_сounterfeit_train.csv
    ml_ozon_сounterfeit_test.csv
  ml_ozon_сounterfeit_train_images/
    *.png
  ml_ozon_сounterfeit_test_images/
    *.png
```

Images are matched to tabular records by `ItemID`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Build the manifest and create the data splits:

```bash
python3 scripts/build_manifest.py
```

Run data checks:

```bash
python3 scripts/run_data_checks.py
```

Train the baseline models:

```bash
python3 scripts/train_experiment.py --experiment e0
python3 scripts/train_experiment.py --experiment e1
python3 scripts/train_experiment.py --experiment e2
```

Compile the metrics table:

```bash
python3 scripts/collect_results.py
```

Generate PR-AUC comparison plots:

```bash
python3 scripts/plot_pr_auc_comparisons.py
```

## Results

Summary of results on the internal test split:

| Model | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Tabular only | 0.225 | 0.323 | 0.311 | 0.317 |
| Text only | 0.509 | 0.449 | 0.716 | 0.552 |
| Tabular + text | 0.594 | 0.572 | 0.616 | 0.593 |
| All modalities, early fusion | 0.607 | 0.614 | 0.541 | 0.575 |
| All modalities, gated fusion | 0.663 | 0.645 | 0.563 | 0.601 |
| Final late fusion | 0.630 | 0.622 | 0.577 | 0.599 |

The final late-fusion model was selected based on validation PR-AUC. It achieved a PR-AUC of `0.808` on the validation split and `0.630` on the test split.

The main artifacts are saved under `artifacts/`:

- `artifacts/experiments/` - outputs and metrics from training runs.
- `artifacts/reports/` - final tables and plots.
- `artifacts/predictions/` - predictions for the external test set.

The project evaluates how much combining tabular, text, and image features improves counterfeit product detection compared with using individual modalities.
