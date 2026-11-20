# Lab 13 — End-to-End MLOps Integration Report

## Objective

Integrate data versioning and schema validation, preprocessing, training, model registry, evaluation and quality gates, lifecycle decisions, containerized deployment, multi-modal drift monitoring, retraining triggers, and operational documentation into one traceable MLOps system.

## Integrated Pipeline Architecture

The complete system executes the following unified flow:

1. **Data Ingestion & Lineage**: Lineage verification against `dataset_lineage.json` using SHA-256 batch hashes.
2. **Schema & Output Validation**: Pandera schema checks rejecting missing columns, wrong dtypes, and NaNs.
3. **Reproducible Modeling**: Stratified train/test splits, StandardScaler, and Random Forest baseline with fixed random seeds.
4. **Model Lifecycle Promotion**: Automatic candidate promotion if quality gate ($\text{recall} \ge 0.70$) is satisfied and candidate outperforms the active Production version.
5. **Deployment Packaging**: Automatic staging of verified binary, preprocessor, and metadata into `deployment/`.
6. **Continuous Monitoring**:
   - Data Drift: Kolmogorov-Smirnov test identifying shifts in alcohol, volatile acidity, and sulphates.
   - Prediction Drift: Two-sample KS test on output probabilities.
   - Concept Drift: Degradation tracking across accuracy, precision, recall, and ROC-AUC.
7. **Automated Retraining**: Automated pipeline trigger (`12_monitor_trigger.py`) evaluating incoming batches when distribution drift is detected.
