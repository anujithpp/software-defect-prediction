"""
=============================================================================
  Software Defect Prediction — Academic Mini-Project (6th Semester)
=============================================================================
  Dataset  : NASA PROMISE Repository — jm1.csv
  Goal     : Binary classification — predict whether a software module
             contains a defect (True = buggy, False = clean).

  Pipeline :
    1.  Load              → read jm1.csv into a Pandas DataFrame
    2.  Preprocess        → impute missing values, scale features
    3.  Cross-Validation  → 10-fold Stratified K-Fold (full dataset, pre-split)
    4.  Split             → 80 % training / 20 % testing (stratified)
    5.  Train (baseline)  → Random Forest | Gaussian Naive Bayes | Logistic Regression
    6.  Evaluate          → Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
    7.  Compare           → ranked comparison table (baseline)
    8.  SMOTE             → balance training set; retrain; compare before vs after
    9.  Visualize         → 6 plots saved to plots/ directory + plt.show()

  How to run (from repo root):
    python src/defect_prediction.py
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os       # cross-platform file paths
import sys      # clean error exits

import pandas as pd     # tabular data (like Excel in Python)
import numpy  as np     # fast numeric arrays (used internally by scikit-learn)

# --- Visualization -----------------------------------------------------------
import matplotlib.pyplot as plt   # core plotting library
import seaborn as sns             # statistical plots built on matplotlib

# Apply a clean seaborn theme to every figure produced in this script.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# --- Scikit-learn: Preprocessing & Model Selection ---------------------------
from sklearn.model_selection import (
    train_test_split,   # hold-out split
    StratifiedKFold,    # K-fold that preserves class ratio in each fold
    cross_val_score,    # runs CV and returns per-fold scores
)
from sklearn.preprocessing import StandardScaler   # z-score normalisation
from sklearn.impute        import SimpleImputer     # replaces NaN with column mean

# --- Scikit-learn: Models ----------------------------------------------------
from sklearn.ensemble     import RandomForestClassifier  # ensemble of trees
from sklearn.naive_bayes  import GaussianNB              # Bayesian probabilistic model
from sklearn.linear_model import LogisticRegression      # linear classifier / sigmoid

# --- Scikit-learn: Evaluation ------------------------------------------------
from sklearn.metrics import (
    accuracy_score,        # (TP+TN) / total
    precision_score,       # TP / (TP+FP)
    recall_score,          # TP / (TP+FN)
    f1_score,              # harmonic mean of Precision & Recall
    confusion_matrix,      # 2×2 TP/TN/FP/FN table
    classification_report, # full per-class breakdown
    roc_auc_score,         # Area Under the ROC Curve
    roc_curve,             # TPR vs FPR at all thresholds (for plotting)
)

# --- Imbalanced-learn: SMOTE -------------------------------------------------
# SMOTE = Synthetic Minority Over-sampling Technique
# Creates new synthetic minority-class examples by interpolating between
# existing ones in feature space, addressing class imbalance.
from imblearn.over_sampling import SMOTE


# =============================================================================
# HELPER — PLOTS DIRECTORY
# =============================================================================

def ensure_plots_dir(repo_root: str) -> str:
    """
    Create plots/ under the repo root if it does not already exist.
    Returns the absolute path to plots/.
    """
    plots_dir = os.path.join(repo_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


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
    if not os.path.exists(filepath):
        sys.exit(
            f"\n[ERROR] Dataset not found at: '{filepath}'\n"
            "        Run  python scripts/download_dataset.py  first.\n"
        )

    df = pd.read_csv(filepath)

    print("=" * 62)
    print("  STEP 1 — DATA LOADING")
    print("=" * 62)
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
    df         : pd.DataFrame  — the raw dataset
    target_col : str           — name of the target column (default: 'defects')

    Returns
    -------
    X_scaled      : np.ndarray  — preprocessed feature matrix (n_samples × n_features)
    y             : np.ndarray  — integer-encoded labels (0 = clean, 1 = defective)
    feature_names : list[str]   — names of the retained numeric columns (for plots)
    """
    print("=" * 62)
    print("  STEP 2 — PREPROCESSING")
    print("=" * 62)

    # ── 2a. Separate X (features) from y (target) ────────────────────────
    if target_col not in df.columns:
        sys.exit(f"[ERROR] Column '{target_col}' not found. Check your CSV.")

    X = df.drop(columns=[target_col])   # all columns except 'defects'
    y = df[target_col]                  # only the 'defects' column

    # Keep only numeric columns — drop any stray text/object columns.
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)    # saved for the feature-importance plot

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
    imputer        = SimpleImputer(strategy="mean")
    X_imputed      = imputer.fit_transform(X)
    print(f"  Missing values imputed     : {missing_before}  (strategy = column mean)")

    # ── 2d. Feature scaling: StandardScaler ──────────────────────────────
    # z = (x - μ) / σ    →  zero mean, unit standard deviation
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    print(f"  Feature scaling            : StandardScaler  (µ=0, σ=1)")
    print(f"  Number of features used    : {X_scaled.shape[1]}\n")

    return X_scaled, y.to_numpy(), feature_names


# =============================================================================
# SECTION 3 — 10-FOLD STRATIFIED CROSS-VALIDATION
# -----------------------------------------------------------------------------
# A single train/test split gives ONE performance estimate which can be
# optimistic or pessimistic by chance. Cross-validation (CV) gives a more
# reliable estimate by repeating the evaluation K times on different splits.
#
# How Stratified K-Fold CV works:
#   1. Divide the dataset into K = 10 equal folds.
#   2. In iteration i: train on all folds EXCEPT fold i; test on fold i.
#   3. Repeat for i = 1…10. Every sample is used for testing exactly once.
#   4. Report mean ± std of F1-Score across 10 folds.
#
# "Stratified" — each fold maintains the same class ratio as the full dataset.
# This is critical for imbalanced problems like defect prediction.
#
# NOTE: CV is run on the FULL preprocessed dataset BEFORE the final
# train/test split. Fresh model instances are used so CV is independent
# of the final trained models stored for evaluation.
# =============================================================================

def cross_validate_models(X: np.ndarray, y: np.ndarray, cv: int = 10) -> dict:
    """
    Run stratified K-fold cross-validation for all 3 models.

    Parameters
    ----------
    X  : full preprocessed feature matrix
    y  : full label vector
    cv : number of folds (default 10)

    Returns
    -------
    dict : {model_name: np.ndarray of per-fold F1 scores}
    """
    print("=" * 62)
    print(f"  STEP 3 — {cv}-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 62)
    print("  Run on full preprocessed dataset BEFORE the train/test split.")
    print("  Metric: F1-Score (best single metric for imbalanced classification)\n")

    # Fresh model instances — independent of the models trained in Step 5.
    models_cv = {
        "Random Forest"       : RandomForestClassifier(n_estimators=100,
                                                       random_state=42,
                                                       n_jobs=-1),
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression" : LogisticRegression(C=1.0, max_iter=1000,
                                                   solver="lbfgs",
                                                   random_state=42),
    }

    # StratifiedKFold preserves class ratio in every fold.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    cv_results = {}
    print(f"  {'Model':<28} {'Mean F1':>8} {'Std F1':>7}  {'Min':>6}  {'Max':>6}")
    print("  " + "-" * 58)

    for name, model in models_cv.items():
        # cross_val_score trains + evaluates the model `cv` times.
        # n_jobs=-1 uses all CPU cores to parallelise the folds.
        scores = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name:<28} {scores.mean():>8.4f} {scores.std():>7.4f}"
              f"  {scores.min():>6.4f}  {scores.max():>6.4f}")

    print(f"\n  Mean F1 = average across {cv} folds  (higher is better)")
    print(f"  Std F1  = consistency between folds  (lower is better)\n")

    return cv_results


# =============================================================================
# SECTION 4 — TRAIN / TEST SPLIT
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

    print("=" * 62)
    print("  STEP 4 — TRAIN / TEST SPLIT  (80 % / 20 %)")
    print("=" * 62)
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Testing  samples : {len(X_test):,}")
    print(f"  Stratification   : enabled  (class ratio preserved)\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 5 — MODEL TRAINING
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
#   every other feature given the class label.  In practice this is rarely
#   true, but the classifier still works well.  The "Gaussian" part assumes
#   each feature follows a normal distribution within each class.
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
# SECTION 6 — MODEL EVALUATION
# -----------------------------------------------------------------------------
# Five scalar metrics + confusion matrix are reported for each model.
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
#
#   ROC-AUC    = Area Under the Receiver Operating Characteristic Curve.
#     → The ROC curve plots TPR (Recall) vs FPR (1 - Specificity) at every
#       possible decision threshold (not just 0.5).
#     → AUC summarises the curve as one number:
#         AUC = 0.5 → model has no discriminatory power (random guess)
#         AUC = 1.0 → perfect separation of classes
#     → Threshold-independent: not biased by class imbalance.
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
    dict — all metric values + predicted probabilities (y_prob, for ROC plot)
    """
    # Hard-class predictions (0 or 1) for every test sample.
    y_pred = model.predict(X_test)

    # Soft probabilities: predict_proba returns [P(class=0), P(class=1)].
    # We take index [:, 1] — the probability of the POSITIVE (defective) class.
    # These are used to compute ROC-AUC and to draw the ROC curve.
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Compute all metrics -----------------------------------------------
    accuracy  = accuracy_score (y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score   (y_test, y_pred, zero_division=0)
    f1        = f1_score       (y_test, y_pred, zero_division=0)
    auc       = roc_auc_score  (y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)

    # Unpack the 2×2 confusion matrix into named variables for clarity.
    TN, FP, FN, TP = cm.ravel()

    # --- Print report -------------------------------------------------------
    sep = "=" * 62
    print(sep)
    print(f"  EVALUATION  —  {model_name}")
    print(sep)

    print(f"  {'Metric':<12}  {'Value':>8}   {'Formula / Notes'}")
    print(f"  {'-'*58}")
    print(f"  {'Accuracy':<12}  {accuracy:>7.4f}   (TP+TN) / total")
    print(f"  {'Precision':<12}  {precision:>7.4f}   TP / (TP+FP)")
    print(f"  {'Recall':<12}  {recall:>7.4f}   TP / (TP+FN)  ← most critical")
    print(f"  {'F1-Score':<12}  {f1:>7.4f}   2×(P×R)/(P+R)")
    print(f"  {'ROC-AUC':<12}  {auc:>7.4f}   threshold-independent discriminability")

    # Confusion matrix (annotated with cost interpretation)
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
        "auc"      : auc,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "y_prob"   : y_prob,   # stored for ROC curve plotting
    }


# =============================================================================
# SECTION 7 — COMPARISON SUMMARY
# -----------------------------------------------------------------------------
# Side-by-side table so you can immediately see which model performs best
# on each metric — essential for your viva discussion.
# =============================================================================

def print_comparison(results: list, title: str = "MODEL COMPARISON") -> None:
    """
    Print a ranked comparison table of all evaluated models.

    Parameters
    ----------
    results : list of dicts returned by evaluate_model()
    title   : header label (useful for baseline vs SMOTE headings)
    """
    # Sort by F1-Score descending — best model first.
    results_sorted = sorted(results, key=lambda r: r["f1"], reverse=True)

    sep = "=" * 76
    print(sep)
    print(f"  {title}  (sorted by F1 ↓)")
    print(sep)
    header = (f"  {'Rank':<5} {'Model':<28} "
              f"{'Accuracy':>9} {'Precision':>10} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print(header)
    print("  " + "-" * 72)

    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(results_sorted):
        medal = medals[i] if i < 3 else "   "
        row = (f"  {medal:<5} {r['model']:<28} "
               f"{r['accuracy']:>9.4f} {r['precision']:>10.4f} "
               f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")
        print(row)

    print(sep)

    best = results_sorted[0]
    print(f"\n  Best model (by F1): {best['model']}")
    print(f"  F1={best['f1']:.4f}  |  Accuracy={best['accuracy']*100:.2f} %  "
          f"|  Recall={best['recall']:.4f}  |  AUC={best['auc']:.4f}\n")


# =============================================================================
# SECTION 8 — SMOTE (Synthetic Minority Over-sampling Technique)
# -----------------------------------------------------------------------------
# Problem: jm1 is imbalanced (80.7 % clean vs 19.3 % defective).
# When a model trains on imbalanced data, it is biased toward the majority
# class — it learns to mostly predict "clean" because that minimises total
# error. This produces low Recall (many missed bugs).
#
# SMOTE solution:
#   For each minority (defective) sample, find its K nearest neighbours
#   in the feature space. Create a synthetic sample by:
#     new_sample = x + λ × (x_neighbour - x),  λ ∈ [0, 1] (random)
#   Repeat until minority class count equals majority class count.
#
# CRITICAL RULE — NEVER apply SMOTE to the test set.
#   The test set must reflect the TRUE real-world class distribution.
#   Applying SMOTE there would make performance numbers unrealistically high
#   and give a false impression of how the model behaves in production.
# =============================================================================

def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                random_state: int = 42):
    """
    Balance the training set using SMOTE oversampling.

    Parameters
    ----------
    X_train, y_train : original (imbalanced) training data
    random_state     : seed for reproducibility

    Returns
    -------
    X_resampled, y_resampled : balanced training data (test set NEVER touched)
    """
    print("=" * 62)
    print("  STEP 8 — SMOTE  (Applied to Training Set Only)")
    print("=" * 62)

    before = pd.Series(y_train).value_counts().sort_index()
    print(f"  Training set BEFORE SMOTE:")
    print(f"    Clean (0)     : {before[0]:,}")
    print(f"    Defective (1) : {before[1]:,}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after = pd.Series(y_resampled).value_counts().sort_index()
    print(f"\n  Training set AFTER SMOTE (test set is UNTOUCHED):")
    print(f"    Clean (0)     : {after[0]:,}")
    print(f"    Defective (1) : {after[1]:,}")
    print(f"\n  Synthetic defective examples generated: "
          f"{after[1] - before[1]:,}\n")

    return X_resampled, y_resampled


def print_smote_comparison(before_results: list,
                           after_results: list) -> None:
    """
    Print a before/after SMOTE comparison table for F1 and Recall.

    Parameters
    ----------
    before_results : evaluation dicts from baseline models
    after_results  : evaluation dicts from SMOTE-retrained models
    """
    sep = "=" * 72
    print(sep)
    print("  SMOTE IMPACT — Before vs After Comparison")
    print(sep)
    print(f"  {'Model':<28}  {'F1 (B)':>7}  {'F1 (A)':>7}  {'ΔF1':>8}  "
          f"{'Rcl (B)':>7}  {'Rcl (A)':>7}  {'ΔRcl':>8}")
    print("  " + "-" * 69)

    for b, a in zip(before_results, after_results):
        df1 = a["f1"]     - b["f1"]
        dr  = a["recall"] - b["recall"]
        s1  = "+" if df1 >= 0 else ""
        sr  = "+" if dr  >= 0 else ""
        print(f"  {b['model']:<28}  {b['f1']:>7.4f}  {a['f1']:>7.4f}  "
              f"{s1}{df1:>7.4f}  {b['recall']:>7.4f}  {a['recall']:>7.4f}  "
              f"{sr}{dr:>7.4f}")

    print(sep)
    print()
    print("  VIVA NOTE:")
    print("  SMOTE boosts Recall by exposing the model to more defective")
    print("  patterns during training. The trade-off is often a drop in")
    print("  Precision (more false alarms). F1 balances both effects.")
    print("  Test set is NEVER oversampled — results reflect real-world")
    print("  class distribution (19.3 % defective).\n")


# =============================================================================
# SECTION 9 — VISUALIZATIONS
# -----------------------------------------------------------------------------
# Six plots are generated, each saved as a PNG to plots/ AND shown via
# plt.show() for interactive viewing.
#
#   Plot 1 : Class distribution bar chart
#   Plot 2 : Feature correlation heatmap  (seaborn)
#   Plot 3 : Confusion matrix heatmaps — all 3 models side-by-side
#   Plot 4 : Feature importance bar chart — Random Forest (top 15)
#   Plot 5 : ROC-AUC curves — all 3 models overlaid
#   Plot 6 : SMOTE before/after comparison — F1 and Recall (grouped bars)
# =============================================================================

def _save_and_show(fig: plt.Figure, path: str, label: str) -> None:
    """
    Internal helper: save figure to disk, call plt.show(), then close.
    """
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  {label} Saved → {path}")


# ── Plot 1: Class Distribution ───────────────────────────────────────────────

def plot_class_distribution(y: np.ndarray, plots_dir: str) -> None:
    """
    Bar chart showing count and percentage of clean vs defective modules.
    Visually communicates the class imbalance — a core issue in this project.
    """
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Clean (0)", "Defective (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    total  = len(y)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors,
                  edgecolor="black", width=0.5, alpha=0.88)

    # Annotate bars with count and percentage.
    for bar, count in zip(bars, counts.values):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 60,
                f"{count:,}\n({pct:.1f} %)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_title("Class Distribution — jm1 Dataset", fontsize=14,
                 fontweight="bold")
    ax.set_ylabel("Number of Software Modules", fontsize=12)
    ax.set_xlabel("Class Label", fontsize=12)
    ax.set_ylim(0, max(counts.values) * 1.25)

    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "01_class_distribution.png"),
                   "[Plot 1]")


# ── Plot 2: Correlation Heatmap ──────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, target_col: str,
                             plots_dir: str) -> None:
    """
    Seaborn heatmap of pairwise Pearson correlations between all 21 features.

    Pearson correlation r ∈ [-1, 1]:
      r =  1  → perfect positive linear relationship
      r =  0  → no linear relationship
      r = -1  → perfect negative linear relationship

    High correlations between features indicate multicollinearity. This is
    important context for Logistic Regression (redundant features can inflate
    coefficient variance) and motivates potential future feature selection.
    Only the lower triangle is shown — the matrix is symmetric.
    """
    X_df = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    corr = X_df.corr()

    # mask=upper triangle avoids redundant cells.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.4,
        annot_kws={"size": 7}, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Heatmap (Lower Triangle  |  Pearson r)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "02_correlation_heatmap.png"),
                   "[Plot 2]")


# ── Plot 3: Confusion Matrix Heatmaps ────────────────────────────────────────

def plot_confusion_matrices(results: list, plots_dir: str,
                            suptitle: str = "Confusion Matrices — All 3 Models (Baseline)") -> None:
    """
    Side-by-side seaborn confusion matrix heatmaps for all 3 models.
    Cell values are raw counts. Colour intensity shows relative magnitude.
    Subtitle shows F1 and AUC for quick cross-model comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, r in zip(axes, results):
        cm_arr = np.array([[r["TN"], r["FP"]],
                           [r["FN"], r["TP"]]])
        sns.heatmap(
            cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred Clean", "Pred Defective"],
            yticklabels=["Actual Clean", "Actual Defective"],
            cbar=False, linewidths=1, linecolor="grey"
        )
        ax.set_title(
            f"{r['model']}\nF1 = {r['f1']:.3f}  |  AUC = {r['auc']:.3f}",
            fontsize=11, fontweight="bold"
        )

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "03_confusion_matrices.png"),
                   "[Plot 3]")


# ── Plot 4: Feature Importance ───────────────────────────────────────────────

def plot_feature_importance(rf_model: RandomForestClassifier,
                            feature_names: list,
                            plots_dir: str) -> None:
    """
    Horizontal bar chart of the top 15 Random Forest feature importances.

    Importance = Mean Decrease in Gini Impurity (MDI):
      At each tree node, the chosen feature reduces the weighted impurity
      of the child nodes compared to the parent. Averaged across all nodes
      and all trees, this gives MDI importance. Higher → more informative.

    Note: MDI importance can be biased towards high-cardinality features.
    Permutation importance is an alternative worth mentioning in the viva.
    """
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]   # top 15 descending
    top_names   = [feature_names[i] for i in indices]
    top_values  = importances[indices]

    # Reverse so the most important feature appears at the TOP of the chart.
    top_names  = top_names[::-1]
    top_values = top_values[::-1]

    cmap   = plt.cm.viridis
    colors = cmap(np.linspace(0.3, 0.9, len(top_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_names, top_values, color=colors,
                   edgecolor="black", alpha=0.85)

    # Annotate each bar with its numeric importance value.
    for bar, val in zip(bars, top_values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f" {val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Feature Importance  (Mean Decrease in Gini Impurity)",
                  fontsize=11)
    ax.set_title("Top 15 Feature Importances — Random Forest",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(top_values) * 1.18)
    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "04_feature_importance.png"),
                   "[Plot 4]")


# ── Plot 5: ROC Curves ───────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test: np.ndarray,
                    plots_dir: str) -> None:
    """
    Overlaid ROC curves for all 3 models on a single plot.

    The ROC curve:
      X-axis: False Positive Rate  = FP / (FP + TN) = 1 - Specificity
      Y-axis: True Positive Rate   = TP / (TP + FN) = Recall / Sensitivity

    Each point on the curve corresponds to one decision threshold.
    AUC (Area Under the Curve) summarises the model's discrimination ability
    across ALL thresholds — unlike Accuracy, it is not biased by class imbalance.
    The diagonal dashed line represents a random (50 %) classifier (AUC = 0.5).
    """
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for r, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{r['model']}  (AUC = {r['auc']:.4f})")

    # Random classifier baseline (diagonal).
    ax.plot([0, 1], [0, 1], "k--", lw=1.5,
            label="Random Classifier  (AUC = 0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="grey")

    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate  (Recall / Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — All 3 Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "05_roc_curves.png"),
                   "[Plot 5]")


# ── Plot 6: SMOTE Before/After Comparison ────────────────────────────────────

def plot_smote_comparison(before_results: list, after_results: list,
                          plots_dir: str) -> None:
    """
    Grouped bar chart comparing F1-Score and Recall before vs after SMOTE
    for all 3 models. Two subplots side-by-side. Bars are annotated.
    """
    model_names   = [r["model"] for r in before_results]
    f1_before     = [r["f1"]     for r in before_results]
    f1_after      = [r["f1"]     for r in after_results]
    recall_before = [r["recall"] for r in before_results]
    recall_after  = [r["recall"] for r in after_results]

    x   = np.arange(len(model_names))
    w   = 0.35
    clr_b = "#e74c3c"   # red   — before SMOTE
    clr_a = "#2ecc71"   # green — after SMOTE

    def annotate(ax, bars):
        """Add value labels above each bar."""
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.013, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── F1 subplot -----------------------------------------------------------
    b1 = ax1.bar(x - w/2, f1_before, w, label="Before SMOTE",
                 color=clr_b, alpha=0.85, edgecolor="black")
    b2 = ax1.bar(x + w/2, f1_after,  w, label="After SMOTE",
                 color=clr_a, alpha=0.85, edgecolor="black")
    annotate(ax1, b1)
    annotate(ax1, b2)
    ax1.set_ylabel("F1-Score", fontsize=12)
    ax1.set_title("F1-Score: Before vs After SMOTE",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=12, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # ── Recall subplot -------------------------------------------------------
    b3 = ax2.bar(x - w/2, recall_before, w, label="Before SMOTE",
                 color=clr_b, alpha=0.85, edgecolor="black")
    b4 = ax2.bar(x + w/2, recall_after,  w, label="After SMOTE",
                 color=clr_a, alpha=0.85, edgecolor="black")
    annotate(ax2, b3)
    annotate(ax2, b4)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.set_title("Recall: Before vs After SMOTE",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=12, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Impact of SMOTE on Model Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_and_show(fig, os.path.join(plots_dir, "06_smote_comparison.png"),
                   "[Plot 6]")


# =============================================================================
# MAIN — PIPELINE ORCHESTRATOR
# =============================================================================

def main():
    """
    Runs the complete defect prediction pipeline end-to-end:
      Load → Preprocess → Cross-Validate → Split →
      Train → Evaluate → SMOTE → Retrain → Compare → Visualize
    """
    print("\n" + "=" * 62)
    print("  SOFTWARE DEFECT PREDICTION  —  6th Semester Mini-Project")
    print("=" * 62 + "\n")

    # Resolve repo root so paths work regardless of working directory.
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(REPO_ROOT, "data", "jm1.csv")
    PLOTS_DIR = ensure_plots_dir(REPO_ROOT)
    print(f"  Plots will be saved to: {PLOTS_DIR}\n")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    df = load_data(DATA_PATH)

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    # Now also returns feature_names (needed for the importance plot).
    X, y, feature_names = preprocess_data(df, target_col="defects")

    # ── Step 3: 10-Fold Cross-Validation (on full X, y — before splitting)
    cross_validate_models(X, y, cv=10)

    # ── Step 4: Train/Test Split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 5: Train baseline models ─────────────────────────────────────
    print("=" * 62)
    print("  STEP 5 — BASELINE MODEL TRAINING")
    print("=" * 62)
    rf_model  = train_random_forest(X_train, y_train)
    gnb_model = train_naive_bayes(X_train, y_train)
    lr_model  = train_logistic_regression(X_train, y_train)

    # ── Step 6: Evaluate baseline models ──────────────────────────────────
    print("=" * 62)
    print("  STEP 6 — EVALUATION REPORTS  (Baseline — Before SMOTE)")
    print("=" * 62 + "\n")
    before_results = []
    before_results.append(evaluate_model(rf_model,  X_test, y_test, "Random Forest"))
    before_results.append(evaluate_model(gnb_model, X_test, y_test, "Gaussian Naive Bayes"))
    before_results.append(evaluate_model(lr_model,  X_test, y_test, "Logistic Regression"))

    # ── Step 7: Comparison table (baseline) ───────────────────────────────
    print_comparison(before_results,
                     title="BASELINE MODEL COMPARISON  (Before SMOTE)")

    # ── Step 8: SMOTE — balance training set, retrain, compare ───────────
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    print("=" * 62)
    print("  STEP 8b — MODEL TRAINING  (After SMOTE)")
    print("=" * 62)
    rf_sm  = train_random_forest(X_train_sm, y_train_sm)
    gnb_sm = train_naive_bayes(X_train_sm, y_train_sm)
    lr_sm  = train_logistic_regression(X_train_sm, y_train_sm)

    print("  Evaluating SMOTE models on the original (untouched) test set …\n")
    after_results = []
    after_results.append(evaluate_model(rf_sm,  X_test, y_test, "Random Forest"))
    after_results.append(evaluate_model(gnb_sm, X_test, y_test, "Gaussian Naive Bayes"))
    after_results.append(evaluate_model(lr_sm,  X_test, y_test, "Logistic Regression"))

    print_comparison(after_results,
                     title="MODEL COMPARISON  (After SMOTE)")
    print_smote_comparison(before_results, after_results)

    # ── Step 9: Visualizations ────────────────────────────────────────────
    print("=" * 62)
    print("  STEP 9 — GENERATING VISUALIZATIONS  (6 plots)")
    print("=" * 62)

    plot_class_distribution(y, PLOTS_DIR)
    plot_correlation_heatmap(df, "defects", PLOTS_DIR)
    plot_confusion_matrices(before_results, PLOTS_DIR)
    plot_feature_importance(rf_model, feature_names, PLOTS_DIR)
    plot_roc_curves(before_results, y_test, PLOTS_DIR)
    plot_smote_comparison(before_results, after_results, PLOTS_DIR)

    print(f"\n  All 6 plots saved to: {PLOTS_DIR}")
    print("\n  [DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
