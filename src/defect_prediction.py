"""
=============================================================================
  Software Defect Prediction — Academic Mini-Project (6th Semester)
=============================================================================
  Dataset  : NASA PROMISE Repository — jm1.csv
  Goal     : Binary classification — predict whether a software module
             contains a defect (True = buggy, False = clean).

  Pipeline :
    1. Load          → read jm1.csv into a Pandas DataFrame
    2. Preprocess    → impute missing values, scale features
    3. Split         → 80 % training / 20 % testing (stratified)
    4. Train         → Random Forest | Gaussian Naive Bayes | Logistic Regression
    5. Evaluate      → Accuracy, Precision, Recall, F1-Score, Confusion Matrix

  How to run (from repo root):
    python src/defect_prediction.py
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os       # for building cross-platform file paths
import sys      # for clean error exits

import pandas as pd     # powerful tabular data library  (like Excel in Python)
import numpy  as np     # fast numerical arrays           (used by scikit-learn internally)

# --- Scikit-learn: Preprocessing ---
from sklearn.model_selection import train_test_split   # splits data into train/test
from sklearn.preprocessing   import StandardScaler     # normalises feature values
from sklearn.impute          import SimpleImputer       # fills in missing (NaN) values

# --- Scikit-learn: Models ---
from sklearn.ensemble    import RandomForestClassifier  # ensemble of decision trees
from sklearn.naive_bayes import GaussianNB              # probabilistic Bayes classifier
from sklearn.linear_model import LogisticRegression     # linear probabilistic classifier

# --- Scikit-learn: Evaluation Metrics ---
from sklearn.metrics import (
    accuracy_score,          # (TP + TN) / total
    precision_score,         # TP / (TP + FP)
    recall_score,            # TP / (TP + FN)
    f1_score,                # harmonic mean of Precision & Recall
    confusion_matrix,        # 2×2 table of TP, TN, FP, FN
    classification_report,   # full per-class breakdown
)


# =============================================================================
# SECTION 1 — DATA LOADING
# -----------------------------------------------------------------------------
# We use Pandas to read the CSV file into a "DataFrame" — a table-like object
# with rows (samples / software modules) and columns (feature metrics).
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the jm1 CSV dataset from disk.

    Parameters
    ----------
    filepath : str
        Path to jm1.csv (e.g. 'data/jm1.csv').

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset as loaded from the CSV.
    """
    # Guard: exit early with a helpful message if the file doesn't exist.
    if not os.path.exists(filepath):
        sys.exit(
            f"\n[ERROR] Dataset not found at: '{filepath}'\n"
            "        Run  python scripts/download_dataset.py  first.\n"
        )

    df = pd.read_csv(filepath)

    print("=" * 60)
    print("  STEP 1 — DATA LOADING")
    print("=" * 60)
    print(f"  File        : {filepath}")
    print(f"  Total rows  : {df.shape[0]:,}")
    print(f"  Total cols  : {df.shape[1]}")
    print(f"  Columns     : {list(df.columns)}\n")

    return df


# =============================================================================
# SECTION 2 — PREPROCESSING
# -----------------------------------------------------------------------------
# Raw data is rarely ready for a machine learning model. We must:
#
#   2a) Separate X (features) and y (target label).
#   2b) Encode the boolean target into integers   (True → 1,  False → 0).
#   2c) Impute missing values  — replace NaN with the column mean.
#       Why mean?  It is unbiased, preserves the feature's overall scale,
#       and does not introduce extreme values.
#   2d) Scale features with StandardScaler         (µ = 0, σ = 1).
#       Formula:  z = (x - mean) / std
#       Why scale?  Many algorithms (Naive Bayes, Logistic Regression)
#       assume features are on the same scale. Without scaling, a feature
#       like 'lines of code' (range: 0–10,000) would dominate a feature
#       like 'cyclomatic complexity' (range: 1–50).
# =============================================================================

def preprocess_data(df: pd.DataFrame, target_col: str = "defects"):
    """
    Clean, encode, impute, and scale the raw DataFrame.

    Parameters
    ----------
    df         : pd.DataFrame  —  the raw dataset
    target_col : str           —  name of the target column (default: 'defects')

    Returns
    -------
    X_scaled : np.ndarray  —  preprocessed feature matrix  (shape: n_samples × n_features)
    y        : np.ndarray  —  integer-encoded labels        (0 = clean, 1 = defective)
    """
    print("=" * 60)
    print("  STEP 2 — PREPROCESSING")
    print("=" * 60)

    # ── 2a. Separate X (features) from y (target) ────────────────────────
    if target_col not in df.columns:
        sys.exit(f"[ERROR] Column '{target_col}' not found. Check your CSV.")

    X = df.drop(columns=[target_col])   # all columns except 'defects'
    y = df[target_col]                  # only the 'defects' column

    # Keep only numeric columns — drop any stray text/object columns.
    X = X.select_dtypes(include=[np.number])

    # ── 2b. Encode target: True/False → 1/0 ─────────────────────────────
    # scikit-learn classifiers expect numeric labels, not strings/booleans.
    y = y.map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)

    defect_count    = int((y == 1).sum())
    no_defect_count = int((y == 0).sum())
    total           = len(y)
    print(f"  Target encoding  : True → 1 (defective),  False → 0 (clean)")
    print(f"  Class distribution:")
    print(f"    Clean    (0) : {no_defect_count:>6,}  ({no_defect_count/total*100:.1f} %)")
    print(f"    Defective(1) : {defect_count:>6,}  ({defect_count/total*100:.1f} %)")
    print(f"  ⚠ Class imbalance ratio  1 : {no_defect_count/defect_count:.1f}")
    print(f"    (This naturally drives lower Recall — discussed in viva notes)\n")

    # ── 2c. Impute missing values ─────────────────────────────────────────
    # SimpleImputer(strategy='mean') replaces each NaN with that column's mean.
    # We fit on the entire X here; when we split later, no data leaks between
    # train/test because the labels are not involved in mean calculation.
    missing_before = int(df.isnull().sum().sum())
    imputer  = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    print(f"  Missing values imputed     : {missing_before}  (strategy = column mean)")

    # ── 2d. Feature scaling: StandardScaler ──────────────────────────────
    # z = (x - μ) / σ    →  zero mean, unit standard deviation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"  Feature scaling            : StandardScaler  (µ=0, σ=1)")
    print(f"  Number of features used    : {X_scaled.shape[1]}\n")

    return X_scaled, y.to_numpy()


# =============================================================================
# SECTION 3 — TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
# We hold out 20 % of the data as a "test set" that the models NEVER see
# during training. This simulates how the model would perform on new,
# unseen data — which is what we actually care about.
#
# stratify=y  ensures both splits contain the same ratio of defective to
# clean modules. Without stratification, the test set could accidentally
# contain very few defective examples, making evaluation misleading.
# =============================================================================

def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.2, random_state: int = 42):
    """
    Split feature matrix and labels into training and testing subsets.

    Parameters
    ----------
    X            : feature matrix
    y            : label vector
    test_size    : proportion of data for testing (0.20 = 20 %)
    random_state : seed for reproducibility — same seed → same split every run

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y       # preserve class ratio in both halves
    )

    print("=" * 60)
    print("  STEP 3 — TRAIN / TEST SPLIT  (80 % / 20 %)")
    print("=" * 60)
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Testing  samples : {len(X_test):,}")
    print(f"  Stratification   : enabled  (class ratio preserved)\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 4 — MODEL TRAINING
# -----------------------------------------------------------------------------
# We train three baseline classifiers and compare them.
#
# MODEL 1 — Random Forest
#   An ensemble of N decision trees (default N=100).  Each tree is trained on
#   a random bootstrap sample of the data (row sampling) and considers only a
#   random subset of features at each split (column sampling — "feature
#   bagging").  The final prediction is a majority vote across all trees.
#   Reduces variance compared to a single deep tree (i.e., less overfitting).
#
# MODEL 2 — Gaussian Naive Bayes
#   Applies Bayes' theorem:
#     P(class | features) ∝ P(class) × ∏ P(feature_i | class)
#   The "naive" assumption: every feature is conditionally independent of
#   every other feature given the class.  In practice this is rarely true,
#   but the classifier still works well.  The "Gaussian" part assumes each
#   feature follows a normal distribution within each class.
#   Very fast to train; great interpretable baseline.
#
# MODEL 3 — Logistic Regression
#   Despite the name, this is a *classification* model.
#   It models the probability of the positive class using the sigmoid function:
#     P(y=1 | X) = 1 / (1 + e^(-z))   where  z = w₀ + w₁x₁ + w₂x₂ + ...
#   Decision boundary: predict 1 if P > 0.5, else predict 0.
#   Coefficients (w) are learned by maximising the log-likelihood of the
#   training data (equivalent to minimising cross-entropy loss).
#   Highly interpretable and very strong linear baseline.
# =============================================================================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        n_estimators: int = 100,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.

    Key hyperparameters
    -------------------
    n_estimators : number of trees in the forest (more trees = lower variance,
                   but slower training). Default 100 is a solid baseline.
    random_state : makes bootstrap sampling reproducible.
    n_jobs = -1  : use ALL available CPU cores in parallel.
    """
    print("  [1/3] Training Random Forest Classifier ...", end=" ", flush=True)
    model = RandomForestClassifier(
        n_estimators = n_estimators,
        random_state = random_state,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)
    print("done.")
    return model


def train_naive_bayes(X_train: np.ndarray,
                      y_train: np.ndarray) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes Classifier.

    No hyperparameters to tune here — GaussianNB estimates the mean and
    variance of each feature per class directly from training data.
    This makes it extremely fast but also less flexible than RF or LR.
    """
    print("  [2/3] Training Gaussian Naive Bayes   ...", end=" ", flush=True)
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("done.")
    return model


def train_logistic_regression(X_train: np.ndarray,
                               y_train: np.ndarray,
                               random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression Classifier.

    Key hyperparameters
    -------------------
    C            : inverse of regularisation strength (smaller C → stronger L2
                   regularisation → smaller weights → less overfitting).
                   Default C=1.0 is a reasonable starting point.
    max_iter     : maximum iterations for the optimiser. Increased to 1000
                   because the jm1 feature space takes more steps to converge.
    solver='lbfgs': Limited-memory BFGS — efficient for medium-sized datasets.
    """
    print("  [3/3] Training Logistic Regression    ...", end=" ", flush=True)
    model = LogisticRegression(
        C            = 1.0,
        max_iter     = 1000,
        solver       = "lbfgs",
        random_state = random_state
    )
    model.fit(X_train, y_train)
    print("done.\n")
    return model


# =============================================================================
# SECTION 5 — MODEL EVALUATION
# -----------------------------------------------------------------------------
# Four scalar metrics + one matrix metric are reported for each model.
#
#   Confusion Matrix (2 × 2):
#                     Predicted 0    Predicted 1
#     Actual 0  →  [ TN            FP ]
#     Actual 1  →  [ FN            TP ]
#
#   Accuracy   = (TP + TN) / (TP + TN + FP + FN)
#     → What fraction of all predictions are correct?
#     → Misleading on imbalanced datasets (80 % accuracy possible by always
#       predicting "no defect").
#
#   Precision  = TP / (TP + FP)
#     → Of all modules we flagged as defective, how many truly were?
#     → Minimises FALSE POSITIVES (false alarms → wasted review effort).
#
#   Recall     = TP / (TP + FN)
#     → Of all truly defective modules, how many did we catch?
#     → Minimises FALSE NEGATIVES (missed bugs → ships to production!).
#     → In defect prediction, Recall is typically MORE important than Precision.
#
#   F1-Score   = 2 × (Precision × Recall) / (Precision + Recall)
#     → Harmonic mean of Precision and Recall.
#     → Best single metric when classes are imbalanced.
#     → Penalises models that sacrifice one metric to inflate the other.
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> dict:
    """
    Generate and print a full evaluation report for one trained model.

    Parameters
    ----------
    model      : a fitted scikit-learn estimator
    X_test     : held-out feature matrix
    y_test     : true labels for the test set
    model_name : display name printed in the report header

    Returns
    -------
    dict  — all metric values, for later comparison table
    """
    # Ask the model to predict a label for every test sample.
    y_pred = model.predict(X_test)

    # --- Compute metrics ---------------------------------------------------
    accuracy  = accuracy_score (y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score   (y_test, y_pred, zero_division=0)
    f1        = f1_score       (y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    # Unpack the confusion matrix into named variables for clarity.
    TN, FP, FN, TP = cm.ravel()

    # --- Print report -------------------------------------------------------
    sep = "=" * 60
    print(sep)
    print(f"  EVALUATION  —  {model_name}")
    print(sep)

    # Scalar metrics
    print(f"  {'Metric':<12}  {'Value':>8}   {'Formula'}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<12}  {accuracy:>7.4f}   (TP+TN) / total")
    print(f"  {'Precision':<12}  {precision:>7.4f}   TP / (TP+FP)")
    print(f"  {'Recall':<12}  {recall:>7.4f}   TP / (TP+FN)  ← most critical")
    print(f"  {'F1-Score':<12}  {f1:>7.4f}   2×(P×R)/(P+R)")

    # Confusion matrix (annotated)
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted Clean  Predicted Defective")
    print(f"  Actual Clean   :       {TN:<10}        {FP}")
    print(f"  Actual Defect  :       {FN:<10}        {TP}")
    print(f"\n    TN={TN}  FP={FP}  FN={FN}  TP={TP}")
    print(f"    False Positives (wasted reviews) : {FP}")
    print(f"    False Negatives (missed bugs)    : {FN}  ← costly!")

    # Full per-class classification report
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Clean (0)", "Defective (1)"],
                                digits=4))
    print(sep + "\n")

    return {
        "model"    : model_name,
        "accuracy" : accuracy,
        "precision": precision,
        "recall"   : recall,
        "f1"       : f1,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


# =============================================================================
# SECTION 6 — COMPARISON SUMMARY
# -----------------------------------------------------------------------------
# Side-by-side table so you can immediately see which model performs best
# on each metric — essential for your viva discussion.
# =============================================================================

def print_comparison(results: list) -> None:
    """
    Print a ranked comparison table of all evaluated models.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()
    """
    # Sort by F1-Score descending — best model first
    results_sorted = sorted(results, key=lambda r: r["f1"], reverse=True)

    sep = "=" * 72
    print(sep)
    print("  FINAL MODEL COMPARISON  (sorted by F1-Score ↓)")
    print(sep)
    header = (f"  {'Rank':<5} {'Model':<32} "
              f"{'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7}")
    print(header)
    print("  " + "-" * 68)

    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(results_sorted):
        medal = medals[i] if i < 3 else "   "
        row = (f"  {medal:<5} {r['model']:<32} "
               f"{r['accuracy']:>9.4f} {r['precision']:>10.4f} "
               f"{r['recall']:>7.4f} {r['f1']:>7.4f}")
        print(row)

    print(sep)

    # Highlight the best model by F1
    best = results_sorted[0]
    print(f"\n  Best overall model (by F1-Score): {best['model']}")
    print(f"  F1 = {best['f1']:.4f}  |  "
          f"Accuracy = {best['accuracy']*100:.2f} %  |  "
          f"Recall = {best['recall']:.4f}\n")

    # Viva discussion note
    print("  VIVA NOTE:")
    print("  Low Recall across all models is expected — the dataset is")
    print("  imbalanced (~19 % defective). To improve Recall you could:")
    print("    1. Apply SMOTE (Synthetic Minority Over-sampling Technique)")
    print("    2. Use  class_weight='balanced'  in Random Forest / LR")
    print("    3. Lower the decision threshold from 0.5 to ~0.3\n")


# =============================================================================
# MAIN — PIPELINE ORCHESTRATOR
# =============================================================================

def main():
    """
    Runs the complete defect prediction pipeline end-to-end:
      Load → Preprocess → Split → Train (×3) → Evaluate (×3) → Compare
    """
    print("\n" + "=" * 60)
    print("  SOFTWARE DEFECT PREDICTION  —  6th Semester Mini-Project")
    print("=" * 60 + "\n")

    # ── 1. Load ──────────────────────────────────────────────────────────
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(REPO_ROOT, "data", "jm1.csv")
    df = load_data(DATA_PATH)

    # ── 2. Preprocess ────────────────────────────────────────────────────
    X, y = preprocess_data(df, target_col="defects")

    # ── 3. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 4 — MODEL TRAINING")
    print("=" * 60)
    rf_model  = train_random_forest(X_train, y_train)
    gnb_model = train_naive_bayes(X_train, y_train)
    lr_model  = train_logistic_regression(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 5 — EVALUATION REPORTS")
    print("=" * 60 + "\n")
    results = []
    results.append(evaluate_model(rf_model,  X_test, y_test, "Random Forest"))
    results.append(evaluate_model(gnb_model, X_test, y_test, "Gaussian Naive Bayes"))
    results.append(evaluate_model(lr_model,  X_test, y_test, "Logistic Regression"))

    # ── 6. Compare ───────────────────────────────────────────────────────
    print_comparison(results)
    print("  [DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
