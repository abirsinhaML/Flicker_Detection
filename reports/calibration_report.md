# Threshold Calibration Report

The reference set was split deterministically by SHA-256 video key, with 80% used for fitting and 20% held out for evaluation. Threshold pairs were selected by training macro-F1 over score-derived candidate values.

## Final boundaries

- None: score < 0.3365
- Mild: 0.3365 ≤ score < 0.4666
- Extreme: score ≥ 0.4666

## Held-out evaluation

- Training videos: 21
- Validation videos: 4
- Accuracy: 1.000
- Macro F1: 0.667

## Confusion matrix

Rows are reference labels; columns are predicted bands.

| Actual / Predicted | none | mild | extreme |
| --- | ---: | ---: | ---: |
| none | 0 | 0 | 0 |
| mild | 0 | 3 | 0 |
| extreme | 0 | 0 | 1 |
