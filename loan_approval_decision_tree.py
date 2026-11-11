import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; remove to display plots interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
N_SAMPLES = 200
TEST_SIZE = 0.2
FINAL_MAX_DEPTH = 3
DEPTHS_TO_COMPARE = [1, 2, 3, 4, 5, 8, None]
APPROVAL_THRESHOLD = 19

FEATURES = ["monthly_income", "credit_score", "loan_amount", "existing_emi"]
TARGET = "approved"
CLASS_NAMES = ["Rejected", "Approved"]

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create synthetic loan applications.

    Units: monthly_income and existing_emi in thousand rupees,
    loan_amount in lakhs. Approval follows a hidden scoring rule plus noise,
    so the data is realistic but not perfectly separable.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "monthly_income": rng.integers(20, 150, n_samples),
            "credit_score": rng.integers(550, 850, n_samples),
            "loan_amount": rng.integers(2, 50, n_samples),
            "existing_emi": rng.integers(0, 40, n_samples),
        }
    )
    risk_score = (
        (df["credit_score"] - 550) / 300 * 50
        + df["monthly_income"] / 150 * 30
        - df["loan_amount"] / 50 * 25
        - df["existing_emi"] / 40 * 20
        + rng.normal(0, 6, n_samples)
    )
    df[TARGET] = (risk_score > APPROVAL_THRESHOLD).astype(int)
    return df


# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
def compare_depths(X_train, X_test, y_train, y_test) -> list[dict]:
    """Train one tree per max_depth value and record train/test accuracy."""
    results = []
    for depth in DEPTHS_TO_COMPARE:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        results.append(
            {
                "max_depth": depth,
                "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
                "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
            }
        )
    return results


def train_final_model(X_train, y_train) -> DecisionTreeClassifier:
    """Train the final depth-limited tree."""
    model = DecisionTreeClassifier(
        criterion="gini", max_depth=FINAL_MAX_DEPTH, random_state=RANDOM_STATE
    )
    return model.fit(X_train, y_train)


def evaluate(model, X_test, y_test) -> dict:
    """Return accuracy, confusion matrix and per-class metrics."""
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    logger.info("Classification report:\n%s", classification_report(
        y_test, y_pred, target_names=CLASS_NAMES))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "recall_rejected": report["Rejected"]["recall"],
        "recall_approved": report["Approved"]["recall"],
    }


def predict_applicant(model, applicant: dict) -> dict:
    """Predict approval for a single applicant."""
    row = pd.DataFrame([applicant])[FEATURES]
    label = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0][label])
    return {"applicant": applicant, "prediction": CLASS_NAMES[label], "confidence": confidence}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def save_tree_plot(model, path: Path) -> None:
    plt.figure(figsize=(18, 9))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title(f"Loan Approval Decision Tree (max_depth={FINAL_MAX_DEPTH})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_importance_plot(importance: pd.Series, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    importance.sort_values().plot(kind="barh", color="#2E75B6")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    df = generate_dataset()
    df.to_csv(BASE_DIR / "loan_data.csv", index=False)
    logger.info("Dataset: %d rows, %d features", len(df), len(FEATURES))
    logger.info("Class balance:\n%s\n", df[TARGET].value_counts().to_string())

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d | Test: %d\n", len(X_train), len(X_test))

    depth_results = compare_depths(X_train, X_test, y_train, y_test)
    logger.info("max_depth | train acc | test acc")
    for r in depth_results:
        logger.info("%9s | %9.3f | %8.3f", r["max_depth"], r["train_accuracy"], r["test_accuracy"])

    model = train_final_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    logger.info("Final model accuracy: %.3f", metrics["accuracy"])
    logger.info("Confusion matrix: %s\n", metrics["confusion_matrix"])

    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    logger.info("Feature importance:\n%s\n", importance.round(3).to_string())

    sample = predict_applicant(
        model,
        {"monthly_income": 80, "credit_score": 740, "loan_amount": 15, "existing_emi": 8},
    )
    logger.info("Sample prediction: %s (confidence %.0f%%)", sample["prediction"], sample["confidence"] * 100)

    save_tree_plot(model, BASE_DIR / "decision_tree.png")
    save_importance_plot(importance, BASE_DIR / "feature_importance.png")

    results = {
        "depth_results": depth_results,
        "final_model": {"max_depth": FINAL_MAX_DEPTH, **metrics},
        "feature_importance": importance.round(4).to_dict(),
        "sample_prediction": sample,
    }
    (BASE_DIR / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("Saved: loan_data.csv, decision_tree.png, feature_importance.png, results.json")


if __name__ == "__main__":
    main()