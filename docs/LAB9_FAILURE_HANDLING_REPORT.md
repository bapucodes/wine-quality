# Lab 9 — Pipeline Failure Handling and Recovery

## Goal

Intentionally break critical stages of the MLOps pipeline and verify that the system halts safely or recovers through the Model Registry.

## Run Locally

```bash
python src/mlops_pipeline/10_failure_injection.py
```

## Scenarios

| Scenario | Injected Fault | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|---|
| **1. Missing Data** | Missing raw dataset (`winequality.csv`) | HALT | HALT | PASSED |
| **2. Invalid Datatype** | Corrupted numeric column with string values | HALT | HALT | PASSED |
| **3. Schema Mismatch** | Dropping required feature column (`pH`) | HALT | HALT | PASSED |
| **4. Preprocessing Corruption** | Injecting NaNs into processed features | HALT | HALT | PASSED |
| **5. Missing Model Artifact** | Deleting local binary (`models/wine_quality_model.pkl`) | RECOVER from Registry | RECOVERED | PASSED |

## Repeated Execution

Executed across 3 repetitions ($5 \text{ scenarios} \times 3 \text{ runs} = 15 \text{ tests}$), achieving a 100% pass rate.
Full evidence saved to `reports/failure_report.json`.
