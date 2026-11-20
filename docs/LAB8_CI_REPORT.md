# Lab 8 — Pipeline Automation and CI Workflows

## Goal

Automate the normal Red Wine Quality Prediction MLOps pipeline with GitHub Actions. Lab 8 automatically executes the core MLOps pipeline when relevant repository changes occur.

## CI Flow

```text
Git push / pull request / manual trigger
                ↓
        install dependencies
                ↓
wine_quality_pipeline.py
                ↓
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
                ↓
        upload reports/artifacts
```

## Triggered Changes

- `config.yaml`
- `src/mlops_pipeline/**`
- `requirements.txt`
- `data/raw/winequality.csv`
- the CI workflow itself (`.github/workflows/lab8_ci.yml`)

## Quality Gate

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

Achieved baseline recall is `0.7670`, successfully meeting and exceeding the `0.70` gate threshold.
