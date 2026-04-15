"""
=============================================================================
Software Defect Prediction — Academic Mini-Project (6th Semester)
=============================================================================
Dataset  : NASA PROMISE Repository — jm1.csv
Goal     : Binary classification to predict whether a software module
           contains a defect (True) or not (False).
Models   : 1) Random Forest Classifier
           2) Gaussian Naive Bayes Classifier
Metrics  : Accuracy, Precision, Recall, Confusion Matrix
=============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────
import os
import sys

# ── Data handling ─────────────────────────────────────────────────────────
import pandas as pd
import numpy as np

# ── Scikit-learn: Preprocessing ───────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ── Scikit-learn: Models ──────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ── Scikit-learn: Evaluation ──────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the CSV dataset from the given file path.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the jm1.csv file.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame as loaded from the CSV.
    """
    if not os.path.exists(filepath):
        sys.exit(f"[ERROR] File not found: '{filepath}'. "
                 "Please place jm1.csv in the project directory.")

    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded successfully.")
    print(f"       Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"       Columns: {list(df.columns)}\n")
    return df


# =============================================================================
# SECTION 2 — PREPROCESSING
# =============================================================================

def preprocess_data(df: pd.DataFrame, target_col: str = "defects"):
    """
    Perform all preprocessing steps:
      1. Separate features (X) from the target label (y).
      2. Convert boolean target values to integers (True → 1, False → 0).
      3. Impute missing values using the column mean (SimpleImputer).
      4. Standardise features using StandardScaler (zero mean, unit variance).

    Parameters
    ----------
    df         : pd.DataFrame — The raw dataset.
    target_col : str          — Name of the target column (default: 'defects').

    Returns
    -------
    X_scaled : np.ndarray  — Preprocessed feature matrix.
    y        : np.ndarray  — Integer-encoded target vector (0 or 1).
    """
    # ── 2a. Split features and target ────────────────────────────────────
    if target_col not in df.columns:
        sys.exit(f"[ERROR] Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── 2b. Encode boolean target → integer ───────────────────────────────
    # The 'defects' column contains Python bool strings ("True"/"False").
    # map() converts them to 1/0 for scikit-learn compatibility.
    y = y.map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)

    print(f"[INFO] Target distribution (after encoding):")
    print(f"       No defect (0) : {(y == 0).sum()}")
    print(f"       Defect    (1) : {(y == 1).sum()}\n")

    # Keep only numeric columns (drop any accidental string columns)
    X = X.select_dtypes(include=[np.number])

    # ── 2c. Impute missing values ─────────────────────────────────────────
    # strategy='mean' replaces NaN with the column's arithmetic mean.
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    missing_count = df.isnull().sum().sum()
    print(f"[INFO] Missing values found and imputed: {missing_count}\n")

    # ── 2d. Feature scaling ───────────────────────────────────────────────
    # StandardScaler normalises each feature to have µ=0 and σ=1.
    # This prevents large-magnitude features from dominating the model.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print(f"[INFO] Feature scaling applied (StandardScaler).")
    print(f"       Number of features: {X_scaled.shape[1]}\n")

    return X_scaled, y.to_numpy()


# =============================================================================
# SECTION 3 — TRAIN / TEST SPLIT
# =============================================================================

def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing subsets.

    Parameters
    ----------
    X            : Preprocessed feature matrix.
    y            : Target label vector.
    test_size    : Fraction of data reserved for testing (default: 0.20 → 20%).
    random_state : Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # preserve class ratio in both splits
    )

    print(f"[INFO] Train-Test Split (80/20 with stratification):")
    print(f"       Training samples : {len(X_train)}")
    print(f"       Testing  samples : {len(X_test)}\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 4 — MODEL TRAINING
# =============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 100,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.

    How it works:
        Builds an ensemble of 'n_estimators' decision trees, each trained on
        a random bootstrap sample of the training data. The final prediction
        is determined by majority voting across all trees. This reduces
        variance and is robust to overfitting.

    Parameters
    ----------
    n_estimators : Number of trees in the forest (default: 100).
    random_state : Seed for reproducibility.

    Returns
    -------
    Trained RandomForestClassifier model.
    """
    print("[INFO] Training Random Forest Classifier ...")
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1           # use all CPU cores for speed
    )
    rf_model.fit(X_train, y_train)
    print("       ✔ Training complete.\n")
    return rf_model


def train_naive_bayes(X_train: np.ndarray,
                      y_train: np.ndarray) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes Classifier.

    How it works:
        Applies Bayes' theorem with the 'naive' assumption that every
        feature is conditionally independent of every other feature given
        the class label. The Gaussian variant assumes each feature follows
        a normal (Gaussian) distribution within each class.
        It is fast, interpretable, and works well as a strong baseline.

    Returns
    -------
    Trained GaussianNB model.
    """
    print("[INFO] Training Gaussian Naive Bayes Classifier ...")
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)
    print("       ✔ Training complete.\n")
    return gnb_model


# =============================================================================
# SECTION 5 — MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> dict:
    """
    Evaluate a trained classifier and print a full performance report.

    Metrics reported:
      • Accuracy  — Overall fraction of correct predictions.
      • Precision — Of all predicted defects, how many were real defects?
                    (Minimises false alarms / false positives)
      • Recall    — Of all actual defects, how many did we catch?
                    (Minimises missed defects / false negatives)
      • Confusion Matrix — 2×2 table of TP, TN, FP, FN counts.

    Parameters
    ----------
    model      : A trained scikit-learn classifier.
    X_test     : Test feature matrix.
    y_test     : True test labels.
    model_name : Display name for the model.

    Returns
    -------
    dict : Dictionary containing all computed metric values.
    """
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    separator = "=" * 55
    print(separator)
    print(f"  EVALUATION REPORT — {model_name}")
    print(separator)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"             Predicted 0   Predicted 1")
    print(f"  Actual 0 :    {cm[0][0]:<10}   {cm[0][1]}")
    print(f"  Actual 1 :    {cm[1][0]:<10}   {cm[1][1]}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["No Defect", "Defect"]))
    print(separator + "\n")

    return {
        "model"     : model_name,
        "accuracy"  : accuracy,
        "precision" : precision,
        "recall"    : recall,
        "confusion_matrix": cm,
    }


# =============================================================================
# SECTION 6 — COMPARISON SUMMARY
# =============================================================================

def print_comparison(results: list[dict]) -> None:
    """
    Print a side-by-side comparison table of all evaluated models.

    Parameters
    ----------
    results : List of result dictionaries returned by evaluate_model().
    """
    print("=" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 55)
    header = f"  {'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8}"
    print(header)
    print("  " + "-" * 53)
    for r in results:
        row = (f"  {r['model']:<30} "
               f"{r['accuracy']:>9.4f} "
               f"{r['precision']:>10.4f} "
               f"{r['recall']:>8.4f}")
        print(row)
    print("=" * 55)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Orchestrates the full ML pipeline:
      1. Load data
      2. Preprocess (impute + scale)
      3. Split into train/test sets
      4. Train Random Forest and Naive Bayes models
      5. Evaluate and compare both models
    """
    print("\n" + "=" * 55)
    print("  SOFTWARE DEFECT PREDICTION — Mini Project")
    print("=" * 55 + "\n")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    # Resolves to  <repo-root>/data/jm1.csv
    # __file__ is src/defect_prediction.py  →  .. goes up to repo root.
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(REPO_ROOT, "data", "jm1.csv")
    df = load_data(DATA_PATH)

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    X, y = preprocess_data(df, target_col="defects")

    # ── Step 3: Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 4: Train ─────────────────────────────────────────────────────
    rf_model  = train_random_forest(X_train, y_train)
    gnb_model = train_naive_bayes(X_train, y_train)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    results = []
    results.append(evaluate_model(rf_model,  X_test, y_test,
                                  "Random Forest Classifier"))
    results.append(evaluate_model(gnb_model, X_test, y_test,
                                  "Gaussian Naive Bayes"))

    # ── Step 6: Summary ───────────────────────────────────────────────────
    print_comparison(results)
    print("\n[DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
