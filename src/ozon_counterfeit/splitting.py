from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .constants import DEFAULT_SPLIT_RATIOS, SELLER_ID_COLUMN, TARGET_COLUMN


def greedy_group_split(
    frame: pd.DataFrame,
    group_col: str = SELLER_ID_COLUMN,
    label_col: str = TARGET_COLUMN,
    split_ratios: dict[str, float] | None = None,
    random_state: int = 42,
) -> dict[str, str]:
    split_ratios = split_ratios or DEFAULT_SPLIT_RATIOS
    split_names = list(split_ratios)

    working = frame[[group_col, label_col]].copy()
    working[group_col] = working[group_col].fillna("__missing_seller__").astype(str)
    working[label_col] = working[label_col].astype(int)

    group_stats = (
        working.groupby(group_col)[label_col]
        .agg(total="count", positive="sum")
        .reset_index()
    )
    group_stats["negative"] = group_stats["total"] - group_stats["positive"]
    group_stats["positive_rate"] = group_stats["positive"] / group_stats["total"].clip(lower=1)

    overall_positive_rate = working[label_col].mean()
    group_stats["positive_rate_delta"] = (group_stats["positive_rate"] - overall_positive_rate).abs()
    group_stats = (
        group_stats.sample(frac=1.0, random_state=random_state)
        .sort_values(["total", "positive_rate_delta"], ascending=[False, False])
        .reset_index(drop=True)
    )

    total_rows = float(group_stats["total"].sum())
    total_positive = float(group_stats["positive"].sum())
    target_rows = {name: total_rows * ratio for name, ratio in split_ratios.items()}
    target_positive = {name: total_positive * ratio for name, ratio in split_ratios.items()}

    current_rows = defaultdict(float)
    current_positive = defaultdict(float)
    assignments: dict[str, str] = {}

    def assignment_score(split_name: str, rows: float, positive: float) -> float:
        simulated_rows = current_rows.copy()
        simulated_positive = current_positive.copy()
        simulated_rows[split_name] += rows
        simulated_positive[split_name] += positive

        score = 0.0
        for name in split_names:
            row_target = max(target_rows[name], 1.0)
            pos_target = max(target_positive[name], 1.0)
            score += abs(simulated_rows[name] - target_rows[name]) / row_target
            score += abs(simulated_positive[name] - target_positive[name]) / pos_target

            overflow = max(0.0, simulated_rows[name] - row_target * 1.05)
            if overflow:
                score += overflow / row_target * 2.0

        return score

    for index, row in enumerate(group_stats.itertuples(index=False)):
        if index < len(split_names):
            chosen_split = split_names[index]
        else:
            chosen_split = min(
                split_names,
                key=lambda name: assignment_score(name, row.total, row.positive),
            )

        assignments[str(getattr(row, group_col))] = chosen_split
        current_rows[chosen_split] += row.total
        current_positive[chosen_split] += row.positive

    return assignments


def compute_split_summary(
    frame: pd.DataFrame,
    split_col: str = "split",
    label_col: str = TARGET_COLUMN,
    group_col: str = SELLER_ID_COLUMN,
) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in frame.groupby(split_col):
        positives = int(split_frame[label_col].sum())
        total = int(len(split_frame))
        rows.append(
            {
                "split": split_name,
                "rows": total,
                "positives": positives,
                "negatives": total - positives,
                "positive_rate": positives / total if total else 0.0,
                "unique_sellers": int(split_frame[group_col].nunique()),
                "with_image_rows": int(split_frame["has_image"].sum()) if "has_image" in split_frame else None,
            }
        )

    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)
