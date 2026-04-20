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
    GridSearchCV,       # exhaustive hyperparameter search with cross-validation
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

# --- SciPy: Statistics -------------------------------------------------------
# pointbiserialr computes the point-biserial correlation between a continuous
# feature and a binary target — the right tool for EDA on defect prediction.
from scipy.stats import pointbiserialr


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


def ensure_eda_dir(plots_dir: str) -> str:
    """
    Create plots/eda/ under plots_dir if it does not already exist.
    Returns the absolute path to plots/eda/.
    """
    eda_dir = os.path.join(plots_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    return eda_dir


# =============================================================================
# SECTION 0 — EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
# EDA runs BEFORE the ML pipeline to build intuition about the dataset:
#
#   0a) Basic statistics   — shape, dtypes, missing values, describe()
#   0b) Feature distributions — histograms per feature, colored by class
#   0c) Outlier detection  — IQR-based outlier counts + boxplot of top 10
#   0d) Defect correlation — point-biserial r between each feature & label
#
# All plots are saved to plots/eda/ AND displayed with plt.show().
# =============================================================================

def run_eda(df: pd.DataFrame, plots_dir: str) -> None:
    """
    Run a complete Exploratory Data Analysis on the raw jm1 DataFrame.

    Parameters
    ----------
    df        : pd.DataFrame  — raw dataset as returned by load_data()
    plots_dir : str           — path to the plots/ directory
    """
    print("=" * 62)
    print("  STEP 0 — EXPLORATORY DATA ANALYSIS  (EDA)")
    print("=" * 62 + "\n")

    # Ensure the plots/eda/ sub-directory exists.
    eda_dir = ensure_eda_dir(plots_dir)

    # Separate numeric features from the boolean target.
    TARGET_COL   = "defects"
    X_df         = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
    feature_cols = list(X_df.columns)      # list of the 21 numeric feature names

    # Encode target to 0/1 integers for correlation maths.
    y_raw = df[TARGET_COL].map({True: 1, False: 0, "True": 1, "False": 0}).astype(int)

    # ── 0a. Basic Statistics ─────────────────────────────────────────────────
    print("  ── 0a. Basic Statistics ──")
    print(f"  Dataset shape     : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Feature columns   : {len(feature_cols)}  ({feature_cols[:4]} … )")
    print()

    # Data types
    print("  Column dtypes:")
    print(df.dtypes.to_string(dtype=False))
    print()

    # Missing value report
    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    if len(missing_nonzero) == 0:
        print("  Missing values    : None 🎉")
    else:
        print(f"  Missing values ({len(missing_nonzero)} columns):")
        print(missing_nonzero.to_string())
    print()

    # Descriptive statistics for numeric features
    print("  Descriptive Statistics (numeric features):")
    desc = X_df.describe().T  # transpose so features are rows
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", 30):
        print(desc)
    print()

    # ── 0b. Feature Distributions (histograms colored by class) ─────────────
    print("  ── 0b. Feature Distributions (histograms, overlay by class) ──")

    # Build a combined DataFrame with the integer label so we can split by class.
    df_plot = X_df.copy()
    df_plot["defect"] = y_raw.values

    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))  # e.g. 21 feats → 6 rows

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.5, n_rows * 3.4))
    axes_flat = axes.flatten()

    clr_clean = "#3498db"     # blue  — class 0 (clean)
    clr_defec = "#e74c3c"     # red   — class 1 (defective)

    for idx, feat in enumerate(feature_cols):
        ax = axes_flat[idx]
        # Separate values by class and plot overlapping histograms.
        vals_clean = df_plot.loc[df_plot["defect"] == 0, feat].dropna()
        vals_defec = df_plot.loc[df_plot["defect"] == 1, feat].dropna()

        ax.hist(vals_clean, bins=30, alpha=0.60, color=clr_clean,
                label="Clean (0)",   density=True, edgecolor="none")
        ax.hist(vals_defec, bins=30, alpha=0.65, color=clr_defec,
                label="Defect (1)",  density=True, edgecolor="none")

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

        if idx == 0:                          # legend only on first subplot
            ax.legend(fontsize=7)

    # Hide any unused subplots (last row may have empty cells).
    for ax in axes_flat[len(feature_cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Feature Distributions — Overlay by Defect Class  (density-normalised)",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    dist_path = os.path.join(eda_dir, "feature_distributions.png")
    _save_and_show(fig, dist_path, "[EDA Plot 1]")
    print(f"  Feature distribution grid saved → {dist_path}\n")

    # ── 0c. Outlier Detection  (IQR method) ──────────────────────────────────
    print("  ── 0c. Outlier Detection  (IQR method) ──")
    print("  Points beyond  Q1 − 1.5×IQR  or  Q3 + 1.5×IQR  are outliers.\n")

    outlier_counts = {}     # {feature_name: int count_of_outliers}
    for feat in feature_cols:
        col_data = X_df[feat].dropna()
        Q1  = col_data.quantile(0.25)
        Q3  = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        n_out = int(((col_data < lo) | (col_data > hi)).sum())
        outlier_counts[feat] = n_out

    # Build a ranked Series (most outlier-prone first).
    outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)

    print(f"  {'Feature':<28}  {'Outlier Count':>13}  {'Pct of total':>13}")
    print("  " + "-" * 58)
    for feat, cnt in outlier_series.items():
        pct = cnt / len(df) * 100
        print(f"  {feat:<28}  {cnt:>13,}  {pct:>12.1f} %")
    print()

    # Boxplot of the top 10 most outlier-prone features.
    top10_feats = list(outlier_series.head(10).index)

    fig2, ax2 = plt.subplots(figsize=(13, 6))
    # Standardise (z-score) so wildly different scales don't hide each other.
    from sklearn.preprocessing import StandardScaler as _SS
    X_scaled_eda = _SS().fit_transform(X_df[top10_feats])
    df_box = pd.DataFrame(X_scaled_eda, columns=top10_feats)

    df_box_melt = df_box.melt(var_name="Feature", value_name="Z-score")
    sns.boxplot(
        data=df_box_melt, x="Feature", y="Z-score",
        palette="Set2", linewidth=1.2, fliersize=2, ax=ax2
    )
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_title(
        "Boxplot — Top 10 Most Outlier-Prone Features  (z-score normalised)",
        fontsize=13, fontweight="bold"
    )
    ax2.set_xlabel("Feature", fontsize=11)
    ax2.set_ylabel("Z-score", fontsize=11)
    ax2.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    boxplot_path = os.path.join(eda_dir, "outliers_boxplot.png")
    _save_and_show(fig2, boxplot_path, "[EDA Plot 2]")
    print(f"  Outlier boxplot saved → {boxplot_path}\n")

    # ── 0d. Defect Correlation (point-biserial) ───────────────────────────────
    print("  ── 0d. Defect Correlation  (Point-Biserial r with defect label) ──")
    print("  Point-biserial r = standard Pearson r but one variable is binary.")
    print("  Range: −1 (strong negative) … 0 (none) … +1 (strong positive)\n")

    pb_corr = {}    # {feature: (r, p-value)}
    for feat in feature_cols:
        col_data = X_df[feat].fillna(X_df[feat].mean())   # impute before corr
        r, p = pointbiserialr(col_data, y_raw.values)
        pb_corr[feat] = (r, p)

    # Build a DataFrame sorted by |r| descending.
    pb_df = pd.DataFrame(
        [(f, v[0], v[1]) for f, v in pb_corr.items()],
        columns=["Feature", "r", "p-value"]
    ).assign(abs_r=lambda d: d["r"].abs()) \
     .sort_values("abs_r", ascending=False) \
     .drop(columns="abs_r") \
     .reset_index(drop=True)

    print(f"  {'Rank':<5} {'Feature':<28} {'r':>8}  {'p-value':>12}  Significance")
    print("  " + "-" * 65)
    for rank, row in pb_df.iterrows():
        sig = "***" if row["p-value"] < 0.001 else (
              "**"  if row["p-value"] < 0.01  else (
              "*"   if row["p-value"] < 0.05  else "ns"))
        print(f"  {rank+1:<5} {row['Feature']:<28} {row['r']:>8.4f}  "
              f"{row['p-value']:>12.4e}  {sig}")
    print()

    # Horizontal bar chart — top 15 features by |r|.
    top15 = pb_df.head(15).copy()
    top15 = top15.iloc[::-1]     # flip so highest abs(r) appears at top
    colors_pb = ["#e74c3c" if r > 0 else "#3498db" for r in top15["r"]]

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    bars3 = ax3.barh(top15["Feature"], top15["r"],
                     color=colors_pb, alpha=0.85, edgecolor="black")

    # Annotate each bar with the r value.
    for bar, r_val in zip(bars3, top15["r"]):
        x_pos = r_val + 0.004 if r_val >= 0 else r_val - 0.004
        ha    = "left"         if r_val >= 0 else "right"
        ax3.text(x_pos, bar.get_y() + bar.get_height() / 2,
                 f"{r_val:+.4f}", va="center", ha=ha, fontsize=9)

    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("Point-Biserial Correlation  r  (with defect label)",
                   fontsize=11)
    ax3.set_title(
        "Top 15 Features Correlated with Defects  (Point-Biserial r)",
        fontsize=13, fontweight="bold"
    )
    ax3.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    corr_path = os.path.join(eda_dir, "defect_correlation.png")
    _save_and_show(fig3, corr_path, "[EDA Plot 3]")
    print(f"  Defect correlation chart saved → {corr_path}\n")

    print("  EDA complete.  Results above + 3 plots saved to plots/eda/\n")


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
# SECTION 10 — HYPERPARAMETER TUNING  (Random Forest — GridSearchCV)
# -----------------------------------------------------------------------------
# Default hyperparameters are "good guesses" but rarely optimal. GridSearchCV
# exhaustively trains and evaluates the model for every combination in the
# param_grid, using K-fold stratified cross-validation to score each combo.
#
# Grid size: 3 × 4 × 3 = 36 combinations × 5 folds = 180 model fits.
# Scoring: F1 — best metric for imbalanced classification.
#
# After finding the best params, we retrain on the full training set and
# compare default vs tuned F1 and AUC on the hold-out test set.
# =============================================================================

def tune_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    plots_dir: str
) -> RandomForestClassifier:
    """
    Run GridSearchCV to find optimal Random Forest hyperparameters.

    Parameters
    ----------
    X_train, y_train : training data
    X_test,  y_test  : held-out test data (used for before/after comparison)
    plots_dir        : path to the plots/ directory

    Returns
    -------
    RandomForestClassifier — the BEST model, already fitted on X_train.
    """
    print("=" * 62)
    print("  STEP 10 — HYPERPARAMETER TUNING  (Random Forest, GridSearchCV)")
    print("=" * 62)

    # ── 10a. Evaluate default RF first (for before/after comparison) ─────────
    print("  Training default Random Forest (n_estimators=100) …", end=" ", flush=True)
    rf_default = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_default.fit(X_train, y_train)
    print("done.")

    y_pred_def  = rf_default.predict(X_test)
    y_prob_def  = rf_default.predict_proba(X_test)[:, 1]
    f1_default  = f1_score    (y_test, y_pred_def, zero_division=0)
    auc_default = roc_auc_score(y_test, y_prob_def)

    print(f"  Default  →  F1 = {f1_default:.4f}  |  AUC = {auc_default:.4f}\n")

    # ── 10b. Define search space ─────────────────────────────────────────────
    # 3 × 4 × 3 = 36 unique combinations × 5 folds = 180 total model fits.
    param_grid = {
        "n_estimators"    : [100, 200, 300],
        "max_depth"       : [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
    }

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator  = RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid = param_grid,
        scoring    = "f1",          # optimise for F1 (best for imbalanced data)
        cv         = skf5,          # 5-fold stratified CV inside the search
        n_jobs     = -1,            # parallelise across all CPU cores
        verbose    = 0,             # suppress per-fold noise
        refit      = True           # re-fit the best model on the full X_train
    )

    # ── 10c. Run the search ──────────────────────────────────────────────────
    print("  ⚠ Running GridSearchCV — this may take a few minutes …")
    print(f"  Search space : {len(param_grid['n_estimators'])} n_estimators "
          f"× {len(param_grid['max_depth'])} max_depth "
          f"× {len(param_grid['min_samples_split'])} min_samples_split "
          f"= 36 combinations × 5 folds = 180 fits")
    import time
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  GridSearchCV completed in {elapsed:.1f}s\n")

    # ── 10d. Report best parameters ──────────────────────────────────────────
    best_params = grid_search.best_params_
    best_cv_f1  = grid_search.best_score_
    print("  Best parameters found:")
    for k, v in best_params.items():
        print(f"    {k:<22} : {v}")
    print(f"  Best CV F1 (5-fold)  : {best_cv_f1:.4f}\n")

    # ── 10e. Evaluate the tuned model on the test set ────────────────────────
    rf_tuned   = grid_search.best_estimator_   # already refitted by GridSearchCV
    y_pred_tun = rf_tuned.predict(X_test)
    y_prob_tun = rf_tuned.predict_proba(X_test)[:, 1]
    f1_tuned   = f1_score    (y_test, y_pred_tun, zero_division=0)
    auc_tuned  = roc_auc_score(y_test, y_prob_tun)

    # ── 10f. Before / After table ────────────────────────────────────────────
    sep = "=" * 62
    print(sep)
    print("  HYPERPARAMETER TUNING — Before vs After  (Random Forest)")
    print(sep)
    print(f"  {'Configuration':<28} {'F1':>8}  {'AUC':>8}")
    print("  " + "-" * 48)
    df1 = f1_tuned  - f1_default
    da  = auc_tuned - auc_default
    s1  = "+" if df1 >= 0 else ""
    sa  = "+" if da  >= 0 else ""
    print(f"  {'Default  (n=100, depth=None, mss=2)':<28} {f1_default:>8.4f}  {auc_default:>8.4f}")
    print(f"  {'Tuned    (GridSearchCV best)':<28} {f1_tuned:>8.4f}  {auc_tuned:>8.4f}")
    print(f"  {'Δ (Tuned − Default)':<28} {s1}{df1:>7.4f}  {sa}{da:>7.4f}")
    print(sep + "\n")

    # ── 10g. Save bar chart comparing default vs tuned F1 and AUC ───────────
    plot_hyperparameter_tuning(
        f1_default, f1_tuned, auc_default, auc_tuned, best_params, plots_dir
    )

    return rf_tuned


def plot_hyperparameter_tuning(
    f1_default:  float, f1_tuned:   float,
    auc_default: float, auc_tuned:  float,
    best_params: dict,  plots_dir:  str
) -> None:
    """
    Grouped bar chart: Default vs Tuned Random Forest — F1 and AUC side-by-side.
    Saved to plots/hyperparameter_tuning.png.
    """
    metrics     = ["F1-Score", "ROC-AUC"]
    val_default = [f1_default,  auc_default]
    val_tuned   = [f1_tuned,    auc_tuned]

    x   = np.arange(len(metrics))
    w   = 0.33
    clr_d = "#e74c3c"   # red   — default
    clr_t = "#2ecc71"   # green — tuned

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_d = ax.bar(x - w/2, val_default, w,
                    label="Default (n=100, depth=None, mss=2)",
                    color=clr_d, alpha=0.85, edgecolor="black")
    bars_t = ax.bar(x + w/2, val_tuned,   w,
                    label="Tuned  (GridSearchCV best)",
                    color=clr_t, alpha=0.85, edgecolor="black")

    # Annotate bars.
    for bar in list(bars_d) + list(bars_t):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + 0.005, f"{h:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Build a subtitle showing the best params found.
    param_str = (f"Best params  →  n_estimators={best_params['n_estimators']}  "
                 f"max_depth={best_params['max_depth']}  "
                 f"min_samples_split={best_params['min_samples_split']}")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, min(1.0, max(val_default + val_tuned) * 1.18))
    ax.set_title(
        "Random Forest — Default vs Tuned Hyperparameters\n" + param_str,
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    hp_path = os.path.join(plots_dir, "hyperparameter_tuning.png")
    _save_and_show(fig, hp_path, "[Plot 7]")
    print(f"  Hyperparameter tuning chart saved → {hp_path}\n")


# =============================================================================
# MAIN — PIPELINE ORCHESTRATOR
# =============================================================================

def main():
    """
    Runs the complete defect prediction pipeline end-to-end:
      Load → EDA → Preprocess → Cross-Validate → Split →
      Tune (GridSearchCV) → Train → Evaluate → SMOTE → Retrain → Compare → Visualize
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

    # ── Step 0: EDA (runs right after loading, before preprocessing) ──────
    # Provides visual and statistical understanding of the raw data.
    run_eda(df, PLOTS_DIR)

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    # Now also returns feature_names (needed for the importance plot).
    X, y, feature_names = preprocess_data(df, target_col="defects")

    # ── Step 3: 10-Fold Cross-Validation (on full X, y — before splitting)
    cross_validate_models(X, y, cv=10)

    # ── Step 4: Train/Test Split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ── Step 10: Hyperparameter Tuning — GridSearchCV (Random Forest only) ─
    # Returns the TUNED Random Forest (best params from GridSearchCV),
    # which is then used as the RF model for all downstream evaluation.
    rf_tuned = tune_random_forest(
        X_train, y_train, X_test, y_test, PLOTS_DIR
    )

    # ── Step 5: Train baseline models ─────────────────────────────────────
    # NOTE: Random Forest now uses the GridSearchCV-tuned model.
    print("=" * 62)
    print("  STEP 5 — BASELINE MODEL TRAINING")
    print("=" * 62)
    rf_model  = rf_tuned                                    # tuned RF
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
    # NOTE: Plot 7 (hyperparameter tuning) was already saved in Step 10.
    #       EDA plots (EDA 1-3) were saved in Step 0.
    #       This step generates the remaining 6 core pipeline plots.
    print("=" * 62)
    print("  STEP 9 — GENERATING VISUALIZATIONS  (6 core plots)")
    print("=" * 62)

    plot_class_distribution(y, PLOTS_DIR)
    plot_correlation_heatmap(df, "defects", PLOTS_DIR)
    plot_confusion_matrices(before_results, PLOTS_DIR)
    plot_feature_importance(rf_model, feature_names, PLOTS_DIR)
    plot_roc_curves(before_results, y_test, PLOTS_DIR)
    plot_smote_comparison(before_results, after_results, PLOTS_DIR)

    print(f"\n  Core plots (1-6) saved to    : {PLOTS_DIR}")
    print(f"  Hyperparameter tuning plot   : {os.path.join(PLOTS_DIR, 'hyperparameter_tuning.png')}")
    print(f"  EDA plots (3)                : {os.path.join(PLOTS_DIR, 'eda', '')}")
    print("\n  [DONE] Pipeline complete.\n")


if __name__ == "__main__":
    main()
