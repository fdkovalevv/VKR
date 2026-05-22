from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .constants import DEFAULT_TEXT_FIELDS, EXCLUDED_NUMERIC_COLUMNS, LOG1P_COLUMN_HINTS, TARGET_COLUMN


def build_text_corpus(frame: pd.DataFrame, text_fields: tuple[str, ...] = DEFAULT_TEXT_FIELDS) -> list[str]:
    corpus = []
    for row in frame[list(text_fields)].fillna("").astype(str).itertuples(index=False):
        values = {
            field: value.strip()
            for field, value in zip(text_fields, row, strict=True)
            if value and value.strip()
        }
        text = " [SEP] ".join(f"{field}: {value}" for field, value in values.items())
        corpus.append(text if text else "description: ")
    return corpus


def infer_numeric_columns(frame: pd.DataFrame) -> list[str]:
    numeric_columns = []
    for column in frame.columns:
        if column in EXCLUDED_NUMERIC_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric_columns.append(column)
    return numeric_columns


def infer_log_columns(numeric_columns: list[str]) -> list[str]:
    log_columns = []
    for column in numeric_columns:
        lowered = column.lower()
        if any(hint in lowered for hint in LOG1P_COLUMN_HINTS):
            log_columns.append(column)
    return log_columns


def extract_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET_COLUMN].to_numpy(dtype=np.float32)


@dataclass
class TabularPreprocessor:
    numeric_columns: list[str]
    log_columns: list[str]
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median"))
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted_: bool = False

    def _prepare_numeric(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        numeric_frame = frame[self.numeric_columns].copy()
        missing_flags = numeric_frame.isna().astype(np.float32).to_numpy()

        if self.log_columns:
            for column in self.log_columns:
                numeric_frame[column] = np.log1p(numeric_frame[column].clip(lower=0))

        return numeric_frame, missing_flags

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        numeric_frame, _ = self._prepare_numeric(frame)
        imputed = self.imputer.fit_transform(numeric_frame)
        self.scaler.fit(imputed)
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")

        numeric_frame, missing_flags = self._prepare_numeric(frame)
        imputed = self.imputer.transform(numeric_frame)
        scaled = self.scaler.transform(imputed).astype(np.float32)
        return np.hstack([scaled, missing_flags]).astype(np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    def get_feature_names(self) -> list[str]:
        return self.numeric_columns + [f"{column}__missing" for column in self.numeric_columns]


def build_modality_flags(num_rows: int, image_present: np.ndarray | None = None) -> np.ndarray:
    if image_present is None:
        image_present = np.zeros(num_rows, dtype=np.float32)
    image_present = image_present.astype(np.float32)
    text_present = np.ones(num_rows, dtype=np.float32)
    tabular_present = np.ones(num_rows, dtype=np.float32)
    return np.stack([image_present, text_present, tabular_present], axis=1)
