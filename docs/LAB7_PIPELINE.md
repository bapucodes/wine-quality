# Lab 7 — ML Pipeline Engineering

Lab 7 builds the core Red Wine Quality prediction ML pipeline.

Normal stages introduced in Lab 7:

```text
01 Load Data
02 Raw Data Validation
03 Preprocessing
04 Processed Output Validation
05 Training + Model Registry
06 Model Evaluation
07 Model Quality Gate (recall >= 0.70)
08 Automated Model Lifecycle
09 Production Inference
```

Run the pipeline after the Lab 7 stages are implemented:

```bash
python src/mlops_pipeline/wine_quality_pipeline.py --force
```

Training uses a Random Forest trained on the stratified training split. Evaluation is performed on the held-out test split using accuracy, precision, recall, F1, and ROC-AUC. Each trained model is versioned and registered under:

```text
Wine_Quality_Production_Model
```

The lifecycle uses the standardized MLOps stages:

```text
None → Staging → Production / Archived
```
