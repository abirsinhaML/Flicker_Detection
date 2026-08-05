"""Per-signal discrimination measurement against the reference labels.

Threshold fitting answers "where do the boundaries go"; this answers the prior
question of "does the signal carry the ordering at all". Keeping them apart
matters because a signal can be *anti*-correlated with severity and still admit a
threshold fit that reports a plausible-looking accuracy on a small validation
split. Ranking every signal separately is what exposes that.

Both statistics are rank-based, so they are invariant to how a signal is scaled
or saturated and can be compared across detectors that share no units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LABEL_ORDER = {"none": 0, "mild": 1, "extreme": 2}
DEFAULT_SIGNALS: tuple[str, ...] = (
    "flicker_score",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
    "horizontal_coherence",
)


@dataclass(frozen=True, slots=True)
class SignalDiscrimination:
    """How well one signal orders videos by reference severity."""

    signal: str
    spearman: float
    auc_extreme_vs_none: float
    class_means: dict[str, float]
    class_counts: dict[str, int]

    @property
    def is_inverted(self) -> bool:
        """True when the signal ranks clean footage above artifacted footage."""
        return self.spearman < 0.0


def measure_discrimination(
    flag_manifest_path: str | Path,
    labels_path: str | Path,
    *,
    signals: Sequence[str] = DEFAULT_SIGNALS,
) -> list[SignalDiscrimination]:
    """Rank each available signal against the ordinal reference label.

    Signals absent from the manifest are skipped rather than raising, so an older
    manifest can still be measured on whatever columns it does carry.
    """
    scores = pd.read_csv(flag_manifest_path)
    labels = pd.read_csv(labels_path)
    if "status" not in scores.columns or "video_key" not in scores.columns:
        raise ValueError("Flag manifest must have 'video_key' and 'status' columns")
    if not {"video_key", "label"}.issubset(labels.columns):
        raise ValueError("Labels file must have 'video_key' and 'label' columns")

    merged = scores.loc[scores["status"] == "ok"].merge(
        labels[["video_key", "label"]], on="video_key", how="inner"
    )
    merged["label"] = merged["label"].str.lower().str.strip()
    if unsupported := set(merged["label"]).difference(LABEL_ORDER):
        raise ValueError(f"Unsupported labels: {sorted(unsupported)}")
    if merged.empty:
        raise ValueError("No successful flag rows matched the provided labels")

    severity = merged["label"].map(LABEL_ORDER)
    results = []
    for signal in signals:
        if signal not in merged.columns:
            continue
        values = pd.to_numeric(merged[signal], errors="coerce")
        if values.isna().all():
            continue
        results.append(
            SignalDiscrimination(
                signal=signal,
                spearman=_spearman(values, severity),
                auc_extreme_vs_none=_auc(
                    values[severity == LABEL_ORDER["extreme"]],
                    values[severity == LABEL_ORDER["none"]],
                ),
                class_means={
                    label: float(values[merged["label"] == label].mean())
                    for label in LABEL_ORDER
                    if (merged["label"] == label).any()
                },
                class_counts={
                    label: int((merged["label"] == label).sum()) for label in LABEL_ORDER
                },
            )
        )
    return sorted(results, key=lambda result: -result.auc_extreme_vs_none)


def write_discrimination_report(
    results: Iterable[SignalDiscrimination],
    output_path: str | Path,
) -> None:
    """Write the per-signal ranking table as Markdown."""
    results = list(results)
    if not results:
        raise ValueError("No signals were measured")

    counts = results[0].class_counts
    total = sum(counts.values())
    rows = "\n".join(
        "| `{signal}` | {spearman:+.3f} | {auc:.3f} | {means} |".format(
            signal=result.signal,
            spearman=result.spearman,
            auc=result.auc_extreme_vs_none,
            means=" / ".join(
                f"{result.class_means.get(label, float('nan')):.3f}" for label in LABEL_ORDER
            ),
        )
        for result in results
    )
    inverted = [result.signal for result in results if result.is_inverted]
    warning = (
        "\n> **Inverted signals: "
        + ", ".join(f"`{name}`" for name in inverted)
        + "**. These rank clean footage above artifacted footage, so no threshold\n"
        "> placed on them can separate the bands.\n"
        if inverted
        else ""
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        "# Signal Discrimination Report\n\n"
        f"Measured on {total} reference-labelled videos "
        f"({counts.get('none', 0)} none, {counts.get('mild', 0)} mild, "
        f"{counts.get('extreme', 0)} extreme).\n\n"
        "Spearman is rank correlation with the ordinal label: +1 is perfect\n"
        "agreement, 0 none, negative inverted. AUC separates `extreme` from `none`,\n"
        "where 0.5 is chance. Both are rank-based, so scaling and saturation do not\n"
        "affect them and signals with different units remain comparable.\n\n"
        "| signal | Spearman | AUC extreme vs none | mean none / mild / extreme |\n"
        "| --- | ---: | ---: | --- |\n" + rows + "\n" + warning + "\n"
        "At this sample size a difference of a few hundredths in AUC is not\n"
        "meaningful; only the sign and the broad ordering should be read.\n",
        encoding="utf-8",
    )


def _spearman(values: pd.Series, severity: pd.Series) -> float:
    """Rank correlation, tie-corrected via average ranks."""
    ranked_values = values.rank()
    ranked_severity = severity.rank()
    if ranked_values.std() == 0 or ranked_severity.std() == 0:
        return 0.0
    return float(ranked_values.corr(ranked_severity))


def _auc(positive: pd.Series, negative: pd.Series) -> float:
    """Probability a positive outranks a negative, ties counting a half.

    Equivalent to the Mann-Whitney statistic, computed directly because the pair
    counts here are tiny and clarity beats an extra dependency.
    """
    if positive.empty or negative.empty:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in positive for n in negative)
    return float(wins / (len(positive) * len(negative)))
