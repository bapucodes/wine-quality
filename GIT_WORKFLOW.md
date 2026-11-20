# Git Workflow Documentation — Red Wine Quality MLOps

**Project:** Red Wine Quality Prediction MLOps Pipeline  
**Standard:** Enterprise Git-Based Version Control and Collaborative ML Workflows

This document records the version-control practices and branch policies enforced across the Red Wine Quality MLOps project.

---

## 1. Branching Strategy

The repository follows a trunk-based development strategy with short-lived feature and experimental branches:

- **`main`**: Production-ready code. Protected branch requiring automated CI checks to pass before merging.
- **`feature/modular-scripts`**: Isolation for modularizing preprocessing, training, and evaluation scripts.
- **`feature/drift-monitoring`**: Development of statistical data, prediction, and concept drift analysis modules.
- **`experiment/model-tuning`**: Hyperparameter optimization and algorithm benchmarking branches.

---

## 2. Pull Request & Quality Gate Policy

All pull requests targeting `main` must:
1. Pass raw and output schema validation.
2. Meet the model quality gate: $\text{recall} \ge 0.70$.
3. Pass the automated API and performance pytest suite (18/18 tests passing).
4. Maintain full lineage tracking for any new data commits.
