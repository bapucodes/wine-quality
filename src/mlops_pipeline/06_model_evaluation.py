import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def draw_confusion_matrix_image(cm, output_path="artifacts/confusion_matrix.png"):
    """Render a clean confusion matrix visualization using PIL."""
    width, height = 500, 450
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((120, 25), "Confusion Matrix - Red Wine Quality", fill=(30, 41, 59))

    # Cell values
    tn, fp, fn, tp = cm.ravel()
    cells = [
        ((120, 100, 270, 220), (224, 242, 254), f"TN: {tn}"),
        ((280, 100, 430, 220), (254, 226, 226), f"FP: {fp}"),
        ((120, 230, 270, 350), (254, 226, 226), f"FN: {fn}"),
        ((280, 230, 430, 350), (220, 252, 231), f"TP: {tp}"),
    ]

    for (x1, y1, x2, y2), fill_color, text in cells:
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=(148, 163, 184), width=2)
        draw.text((x1 + 40, y1 + 50), text, fill=(15, 23, 42))

    # Labels
    draw.text((160, 75), "Pred: 0 (Normal)", fill=(71, 85, 105))
    draw.text((320, 75), "Pred: 1 (Good)", fill=(71, 85, 105))
    draw.text((20, 150), "Actual: 0", fill=(71, 85, 105))
    draw.text((20, 280), "Actual: 1", fill=(71, 85, 105))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def draw_roc_curve_image(fpr, tpr, roc_auc, output_path="artifacts/roc_curve.png"):
    """Render a clean ROC curve visualization using PIL."""
    width, height = 500, 450
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Title
    draw.text((110, 25), f"ROC Curve (AUC = {roc_auc:.4f})", fill=(30, 41, 59))

    # Plot bounding box
    ox, oy, pw, ph = 70, 360, 380, 270
    draw.rectangle([ox, oy - ph, ox + pw, oy], outline=(148, 163, 184), width=2)

    # Diagonal random baseline
    draw.line([ox, oy, ox + pw, oy - ph], fill=(148, 163, 184), width=1)

    # Plot ROC curve points
    points = []
    for f, t in zip(fpr, tpr):
        px = ox + int(f * pw)
        py = oy - int(t * ph)
        points.append((px, py))

    if len(points) > 1:
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=(37, 99, 235), width=3)

    # Labels
    draw.text((180, 390), "False Positive Rate", fill=(71, 85, 105))
    draw.text((10, 210), "TPR", fill=(71, 85, 105))
    draw.text((ox, oy + 10), "0.0", fill=(100, 116, 139))
    draw.text((ox + pw - 20, oy + 10), "1.0", fill=(100, 116, 139))
    draw.text((ox - 30, oy - 10), "0.0", fill=(100, 116, 139))
    draw.text((ox - 30, oy - ph), "1.0", fill=(100, 116, 139))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)


def run_evaluation():
    print("[INFO] Starting ML Pipeline Model Evaluation - Wine Quality MLOps...")

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config["paths"]["model"]
    processed_dir = config["data"]["processed_dir"]

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Run 05_train_registry.py first.")
        return False

    try:
        X_test = np.load(os.path.join(processed_dir, "X_test_final.npy"))
        y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    except FileNotFoundError:
        print("[ERROR] Test data not found. Run 03_preprocessing.py first.")
        return False

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\n--- ML Pipeline: Wine Quality Model Evaluation ---")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():12}: {value:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("--------------------------------------------------\n")

    report = {
        "stage": "ML Pipeline - Model Evaluation",
        "evaluated_on": "held-out test split",
        "test_rows": int(len(y_test)),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
        "confusion_matrix": cm.tolist(),
    }

    os.makedirs("reports", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    report_path = config["paths"]["evaluation_report"]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # Generate charts
    draw_confusion_matrix_image(cm, "artifacts/confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    draw_roc_curve_image(fpr, tpr, metrics["roc_auc"], "artifacts/roc_curve.png")

    print(f"[SUCCESS] Evaluation report saved to {report_path}")
    print("[SUCCESS] Confusion matrix saved to artifacts/confusion_matrix.png")
    print("[SUCCESS] ROC curve saved to artifacts/roc_curve.png")
    return True


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
