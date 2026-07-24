"""Threshold fitting and held-out evaluation for reference-labelled videos."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pandas as pd

LABEL_ORDER = {"none": 0, "mild": 1, "extreme": 2}


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Fitted threshold bands and evaluation metrics."""

    mild_threshold: float
    extreme_threshold: float
    train_size: int
    validation_size: int
    validation_accuracy: float
    validation_macro_f1: float
    confusion_matrix: dict[str, dict[str, int]]


def calibrate(
    flag_manifest_path: str | Path,
    labels_path: str | Path,
    *,
    validation_fraction: float = 0.2,
) -> CalibrationResult:
    """Fit score-band thresholds on a deterministic key-level training split."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    scores = pd.read_csv(flag_manifest_path)
    labels = pd.read_csv(labels_path)
    required_scores = {"video_key", "flicker_score", "status"}
    required_labels = {"video_key", "label"}
    if missing := required_scores.difference(scores.columns):
        raise ValueError(f"Flag manifest missing columns: {sorted(missing)}")
    if missing := required_labels.difference(labels.columns):
        raise ValueError(f"Labels file missing columns: {sorted(missing)}")

    merged = scores.loc[scores["status"] == "ok", ["video_key", "flicker_score"]].merge(
        labels[["video_key", "label"]], on="video_key", how="inner"
    )
    merged["label"] = merged["label"].str.lower().str.strip()
    invalid_labels = set(merged["label"]).difference(LABEL_ORDER)
    if invalid_labels:
        raise ValueError(f"Unsupported labels: {sorted(invalid_labels)}")
    if merged.empty:
        raise ValueError("No successful flag rows matched the provided labels")

    validation_mask = merged["video_key"].map(
        lambda key: _is_validation_key(key, validation_fraction)
    )
    train = merged.loc[~validation_mask]
    validation = merged.loc[validation_mask]
    if train.empty or validation.empty:
        raise ValueError("Split produced an empty train or validation partition")

    mild_threshold, extreme_threshold = _fit_thresholds(train)
    predicted = _predict_bands(validation["flicker_score"], mild_threshold, extreme_threshold)
    confusion = _confusion_matrix(validation["label"], predicted)
    return CalibrationResult(
        mild_threshold=mild_threshold,
        extreme_threshold=extreme_threshold,
        train_size=len(train),
        validation_size=len(validation),
        validation_accuracy=_accuracy(confusion),
        validation_macro_f1=_macro_f1(confusion),
        confusion_matrix=confusion,
    )


def write_calibration_report(result: CalibrationResult, output_path: str | Path) -> None:
    """Write the required calibration split, boundaries, and held-out metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {actual} | "
        + " | ".join(str(result.confusion_matrix[actual][predicted]) for predicted in LABEL_ORDER)
        + " |"
        for actual in LABEL_ORDER
    )
    output_path.write_text(
        "# Threshold Calibration Report\n\n"
        "The reference set was split deterministically by SHA-256 video key, with "
        "80% used for fitting and 20% held out for evaluation. Threshold pairs "
        "were selected by training macro-F1 over score-derived candidate values.\n\n"
        "## Final boundaries\n\n"
        f"- None: score < {result.mild_threshold:.4f}\n"
        f"- Mild: {result.mild_threshold:.4f} ≤ score < {result.extreme_threshold:.4f}\n"
        f"- Extreme: score ≥ {result.extreme_threshold:.4f}\n\n"
        "## Held-out evaluation\n\n"
        f"- Training videos: {result.train_size}\n"
        f"- Validation videos: {result.validation_size}\n"
        f"- Accuracy: {result.validation_accuracy:.3f}\n"
        f"- Macro F1: {result.validation_macro_f1:.3f}\n\n"
        "## Confusion matrix\n\n"
        "Rows are reference labels; columns are predicted bands.\n\n"
        "| Actual / Predicted | none | mild | extreme |\n"
        "| --- | ---: | ---: | ---: |\n" + rows + "\n",
        encoding="utf-8",
    )


def _fit_thresholds(train: pd.DataFrame) -> tuple[float, float]:
    candidates = sorted(set(float(score) for score in train["flicker_score"]))
    if len(candidates) < 3:
        raise ValueError("At least three distinct training scores are required")

    best: tuple[float, float, float] | None = None
    for mild_threshold in candidates[:-1]:
        for extreme_threshold in candidates:
            if extreme_threshold <= mild_threshold:
                continue
            predictions = _predict_bands(train["flicker_score"], mild_threshold, extreme_threshold)
            score = _macro_f1(_confusion_matrix(train["label"], predictions))
            candidate = (score, mild_threshold, extreme_threshold)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise ValueError("Could not fit an ordered threshold pair")
    return best[1], best[2]


def _predict_bands(
    scores: Iterable[float], mild_threshold: float, extreme_threshold: float
) -> list[str]:
    return [
        "none" if score < mild_threshold else "mild" if score < extreme_threshold else "extreme"
        for score in scores
    ]


def _confusion_matrix(actual: Iterable[str], predicted: Iterable[str]) -> dict[str, dict[str, int]]:
    matrix = {label: {prediction: 0 for prediction in LABEL_ORDER} for label in LABEL_ORDER}
    for actual_label, predicted_label in zip(actual, predicted, strict=False):
        matrix[actual_label][predicted_label] += 1
    return matrix


def _accuracy(confusion: dict[str, dict[str, int]]) -> float:
    correct = sum(confusion[label][label] for label in LABEL_ORDER)
    total = sum(sum(row.values()) for row in confusion.values())
    return correct / total if total else 0.0


def _macro_f1(confusion: dict[str, dict[str, int]]) -> float:
    scores = []
    for label in LABEL_ORDER:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[actual][label] for actual in LABEL_ORDER if actual != label)
        false_negative = sum(
            confusion[label][predicted] for predicted in LABEL_ORDER if predicted != label
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append((2 * true_positive / denominator) if denominator else 0.0)
    return sum(scores) / len(scores)


def _is_validation_key(video_key: str, validation_fraction: float) -> bool:
    bucket = int(sha256(video_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < validation_fraction
