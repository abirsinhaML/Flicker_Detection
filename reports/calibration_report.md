# Threshold Calibration Report

The reference set was split deterministically by SHA-256 video key, with 80% used for fitting and 20% held out for evaluation. Threshold pairs were selected by training macro-F1 over score-derived candidate values.

## Final boundaries

- None: score < 0.5042
- Mild: 0.5042 ≤ score < 0.5452
- Extreme: score ≥ 0.5452

## Held-out evaluation

- Training videos: 21
- Validation videos: 4
- Accuracy: 0.500
- Macro F1: 0.333

## Confusion matrix

Rows are reference labels; columns are predicted bands.

| Actual / Predicted | none | mild | extreme |
| --- | ---: | ---: | ---: |
| none | 0 | 0 | 0 |
| mild | 0 | 1 | 2 |
| extreme | 0 | 0 | 1 |
