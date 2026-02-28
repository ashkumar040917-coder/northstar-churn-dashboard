"""
================================================================================
NorthStar Retail Bank — Customer Churn Prediction & Retention Value Analysis
================================================================================
Client:      NorthStar Retail Bank (UK) — Fictional Client
Project:     Customer Churn Risk Modelling & Revenue Retention Strategy
Analyst:     [Your Name] | Business Data Analytics
Date:        Q1 2025
Dataset:     Bank Customer Churn Prediction
             https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers
================================================================================
"""

# ── 0. DEPENDENCIES ──────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── GLOBAL STYLE ─────────────────────────────────────────────────────────────
BRAND_BLUE   = "#003865"
BRAND_TEAL   = "#00A499"
BRAND_AMBER  = "#F0A500"
BRAND_RED    = "#C0392B"
BRAND_GREY   = "#6C757D"
PALETTE      = [BRAND_BLUE, BRAND_TEAL, BRAND_AMBER, BRAND_RED, BRAND_GREY]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   12,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "font.family":      "DejaVu Sans",
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "legend.frameon":   False,
})

OUTPUT_DIR = Path("northstar_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name: str, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & INITIAL AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 1: DATA LOADING & INITIAL AUDIT")
print("="*70)

# ── Download instructions (run once) ─────────────────────────────────────────
# kaggle datasets download -d adammaus/predicting-churn-for-bank-customers
# Or download manually from the URL above and place as 'Churn_Modelling.csv'

try:
    df_raw = pd.read_csv("Churn_Modelling.csv")
    print(f"\n  Dataset loaded:  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
except FileNotFoundError:
    print("\n  [INFO] Dataset not found locally — generating representative synthetic data")
    print("         Download from: https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers\n")

    # ── Synthetic dataset mirroring real dataset structure & distributions ──
    rng = np.random.default_rng(42)
    n = 10_000

    geography = rng.choice(["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25])
    gender     = rng.choice(["Male", "Female"], n, p=[0.545, 0.455])
    age        = rng.integers(18, 92, n)
    tenure     = rng.integers(0, 11, n)
    num_prod   = rng.choice([1, 2, 3, 4], n, p=[0.505, 0.459, 0.026, 0.010])
    has_card   = rng.integers(0, 2, n)
    is_active  = rng.integers(0, 2, n)
    credit     = np.clip(rng.normal(650, 97, n).astype(int), 350, 850)
    balance    = np.where(rng.random(n) < 0.36, 0,
                          np.clip(rng.normal(76_485, 62_397, n), 0, 250_000))
    salary     = np.clip(rng.normal(100_090, 57_510, n), 11_500, 199_999)

    # Churn probability driven by key risk factors
    logit = (
        -2.0
        + 0.035 * (age - 40)
        + 0.6   * (geography == "Germany").astype(int)
        - 0.4   * is_active
        + 0.5   * (num_prod >= 3).astype(int)
        - 0.3   * (balance > 0).astype(int)
        - 0.02  * (credit - 650) / 97
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    exited = rng.binomial(1, prob_churn)

    df_raw = pd.DataFrame({
        "RowNumber":       range(1, n + 1),
        "CustomerId":      rng.integers(15_000_000, 16_000_000, n),
        "Surname":         ["Client_" + str(i) for i in range(n)],
        "CreditScore":     credit,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance.round(2),
        "NumOfProducts":   num_prod,
        "HasCrCard":       has_card,
        "IsActiveMember":  is_active,
        "EstimatedSalary": salary.round(2),
        "Exited":          exited,
    })
    print(f"\n  Synthetic dataset created: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

print(f"\n  Churn rate: {df_raw['Exited'].mean():.1%}")
print(f"\n  Schema:\n{df_raw.dtypes.to_string()}")
print(f"\n  Sample:\n{df_raw.head(3).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLEANING & QUALITY ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 2: DATA CLEANING & QUALITY ASSESSMENT")
print("="*70)

df = df_raw.copy()

# Drop non-analytical identifiers
drop_cols = ["RowNumber", "CustomerId", "Surname"]
df.drop(columns=drop_cols, inplace=True)
print(f"\n  Dropped identifier columns: {drop_cols}")

# Missing value audit
missing = df.isnull().sum()
pct_missing = (missing / len(df) * 100).round(2)
quality_report = pd.DataFrame({"Missing_Count": missing, "Missing_%": pct_missing})
print(f"\n  Missing value audit:\n{quality_report[quality_report['Missing_Count'] > 0].to_string() or '  ✓ No missing values detected'}")

# Duplicate check
dupes = df.duplicated().sum()
print(f"\n  Duplicate rows: {dupes}")
df.drop_duplicates(inplace=True)

# Outlier audit — credit score & age
credit_low = (df["CreditScore"] < 300).sum()
age_high   = (df["Age"] > 90).sum()
print(f"\n  Credit score < 300: {credit_low} rows (capped at 300)")
print(f"  Age > 90:           {age_high} rows (retained — plausible)")
df["CreditScore"] = df["CreditScore"].clip(lower=300)

# Negative balance guard
neg_bal = (df["Balance"] < 0).sum()
print(f"  Negative balance:   {neg_bal} rows")

print(f"\n  Clean dataset:  {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── BUSINESS NOTE ─────────────────────────────────────────────────────────────
print("""
  [ANALYST NOTE]
  Dataset is clean with no material data quality issues. The absence of
  missing values is consistent with a CRM extract from a production banking
  system. Identifier columns are removed to prevent data leakage.
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  SECTION 3: FEATURE ENGINEERING")
print("="*70)

# Balance-to-salary ratio — proxy for financial stress / liquidity
df["Balance_Salary_Ratio"] = np.where(
    df["EstimatedSalary"] > 0,
    df["Balance"] / df["EstimatedSalary"],
    0
)

# Age band — behavioural segmentation
df["Age_Band"] = pd.cut(
    df["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30–39", "40–49", "50–59", "60+"]
)

# Tenure band
df["Tenure_Band"] = pd.cut(
    df["Tenure"],
    bins=[-1, 1, 3, 6, 10],
    labels=["0–1 yr", "2–3 yr", "4–6 yr", "7+ yr"]
)

# Zero balance flag — strategic flag for dormant accounts
df["Zero_Balance_Flag"] = (df["Balance"] == 0).astype(int)

# Product engagement score
df["Engagement_Score"] = (
    df["NumOfProducts"]
    + df["HasCrCard"]
    + df["IsActiveMember"]
)

# High-value customer flag (balance > £100k)
df["High_Value_Flag"] = (df["Balance"] > 100_000).astype(int)

# Credit risk tier
df["Credit_Tier"] = pd.cut(
    df["CreditScore"],
    bins=[0, 579, 669, 739, 799, 900],
    labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
)

engineered = ["Balance_Salary_Ratio", "Age_Band", "Tenure_Band",
              "Zero_Balance_Flag", "Engagement_Score", "High_Value_Flag", "Credit_Tier"]
print(f"\n  Engineered {len(engineered)} new features: {engineered}")
print(f"\n  Feature preview:\n{df[engineered].head(3).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 4: EXPLORATORY DATA ANALYSIS")
print("="*70)

churn_rate = df["Exited"].mean()
churn_n    = df["Exited"].sum()
retain_n   = len(df) - churn_n
print(f"\n  Overall churn rate : {churn_rate:.1%}  ({churn_n:,} churned / {len(df):,} total)")

# ── FIG 1: Churn Distribution ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("NorthStar Retail Bank — Customer Churn Overview", fontsize=16, fontweight="bold", color=BRAND_BLUE)

labels = ["Retained", "Churned"]
sizes  = [retain_n, churn_n]
colors = [BRAND_TEAL, BRAND_RED]
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight("bold")
axes[0].set_title("Portfolio Churn Split")

churn_by_country = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
bars = axes[1].bar(churn_by_country.index, churn_by_country.values * 100,
                   color=[BRAND_RED, BRAND_AMBER, BRAND_TEAL])
axes[1].set_title("Churn Rate by Geography")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].axhline(churn_rate * 100, color=BRAND_GREY, linestyle="--", linewidth=1.5, label="Portfolio Avg")
axes[1].legend()
for bar, val in zip(bars, churn_by_country.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.1%}", ha="center", fontsize=10, fontweight="bold")
save_fig("01_churn_overview")

# ── FIG 2: Churn by Key Demographic Segments ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Churn Rate by Customer Segment — NorthStar Bank", fontsize=15, fontweight="bold", color=BRAND_BLUE)

def plot_churn_bar(ax, col, title, palette=None):
    data = df.groupby(col)["Exited"].mean().reset_index()
    data.columns = [col, "ChurnRate"]
    pal = palette or [BRAND_BLUE] * len(data)
    bars = ax.bar(data[col].astype(str), data["ChurnRate"] * 100, color=pal, edgecolor="white", linewidth=0.5)
    ax.axhline(churn_rate * 100, color=BRAND_GREY, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Churn Rate (%)")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                f"{b.get_height():.1f}%", ha="center", fontsize=9)

plot_churn_bar(axes[0, 0], "Age_Band",      "Churn by Age Band",
               [BRAND_TEAL, BRAND_BLUE, BRAND_AMBER, BRAND_RED, "#8B0000"])
plot_churn_bar(axes[0, 1], "NumOfProducts", "Churn by Number of Products",
               [BRAND_TEAL, BRAND_BLUE, BRAND_RED, "#8B0000"])
plot_churn_bar(axes[1, 0], "IsActiveMember","Churn by Activity Status",
               [BRAND_RED, BRAND_TEAL])
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_xticklabels(["Inactive", "Active"])
plot_churn_bar(axes[1, 1], "Gender",        "Churn by Gender",
               [BRAND_BLUE, BRAND_AMBER])
save_fig("02_churn_by_segment")

# ── FIG 3: Distributions — Churned vs Retained ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Feature Distributions: Churned vs Retained Customers", fontsize=14, fontweight="bold", color=BRAND_BLUE)

for ax, col, xlabel in zip(
    axes,
    ["Age", "CreditScore", "Balance"],
    ["Age (years)", "Credit Score", "Account Balance (£)"]
):
    for val, label, color in [(0, "Retained", BRAND_TEAL), (1, "Churned", BRAND_RED)]:
        subset = df[df["Exited"] == val][col]
        ax.hist(subset, bins=30, alpha=0.65, label=label, color=color, density=True, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}" if col == "Balance" else f"{x:,.0f}"))
save_fig("03_distributions_churn")

# ── FIG 4: Correlation Heatmap ────────────────────────────────────────────────
numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary",
                "Balance_Salary_Ratio", "Zero_Balance_Flag", "Engagement_Score",
                "High_Value_Flag", "Exited"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 8}, ax=ax
)
ax.set_title("Correlation Matrix — NorthStar Customer Features", pad=20)
save_fig("04_correlation_heatmap")

# ── PRINT KEY INSIGHTS ────────────────────────────────────────────────────────
churn_age = df.groupby("Age_Band")["Exited"].mean()
churn_geo = df.groupby("Geography")["Exited"].mean()
churn_prod= df.groupby("NumOfProducts")["Exited"].mean()
churn_act = df.groupby("IsActiveMember")["Exited"].mean()

print(f"""
  KEY EDA INSIGHTS:
  ─────────────────────────────────────────────────────────────────────
  1. Geography:   Germany churn = {churn_geo.get('Germany', 0):.1%} vs France {churn_geo.get('France', 0):.1%}
                  (+{(churn_geo.get('Germany',0)-churn_geo.get('France',0))*100:.0f}pp — material market risk)
  2. Age:         Peak churn in 40–59 cohort ({churn_age.get('40–49', 0):.1%} / {churn_age.get('50–59', 0):.1%})
  3. Products:    3–4 product holders show extreme churn (>{churn_prod.get(3, 0):.0%})
                  Likely forced bundling or poor product-market fit
  4. Inactivity:  Inactive members churn at {churn_act.get(0, 0):.1%} vs active at {churn_act.get(1, 0):.1%}
  ─────────────────────────────────────────────────────────────────────
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  SECTION 5: HYPOTHESIS TESTING")
print("="*70)

from scipy import stats

# H1: Churned customers are significantly older than retained
churned_age  = df[df["Exited"] == 1]["Age"]
retained_age = df[df["Exited"] == 0]["Age"]
t_stat, p_val = stats.ttest_ind(churned_age, retained_age)
print(f"""
  H1: Churned customers are significantly older
      Churned mean age  : {churned_age.mean():.1f} yrs
      Retained mean age : {retained_age.mean():.1f} yrs
      t-statistic: {t_stat:.2f}  |  p-value: {p_val:.2e}
      Result: {'REJECT H0 — statistically significant difference' if p_val < 0.05 else 'FAIL TO REJECT H0'}
""")

# H2: Germany customers have higher churn than France (chi-squared)
contingency = pd.crosstab(df["Geography"], df["Exited"])
chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
print(f"""
  H2: Churn rate varies significantly by geography
      Chi-squared: {chi2:.2f}  |  p-value: {p_chi:.2e}  |  DoF: {dof}
      Result: {'REJECT H0 — geography is a significant churn predictor' if p_chi < 0.05 else 'FAIL TO REJECT H0'}
""")

# H3: Inactive members have higher churn (point-biserial correlation)
corr_active, p_active = stats.pointbiserialr(df["IsActiveMember"], df["Exited"])
print(f"""
  H3: Active membership is negatively correlated with churn
      Point-biserial r: {corr_active:.3f}  |  p-value: {p_active:.2e}
      Result: {'REJECT H0 — active membership significantly reduces churn' if p_active < 0.05 else 'FAIL TO REJECT H0'}
""")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DATA PREPARATION FOR MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  SECTION 6: DATA PREPARATION FOR MODELLING")
print("="*70)

df_model = df.copy()

# Encode categoricals
le = LabelEncoder()
for col in ["Geography", "Gender"]:
    df_model[col] = le.fit_transform(df_model[col])

# Drop ordinal feature-engineered bands (already numerically captured)
drop_eng = ["Age_Band", "Tenure_Band", "Credit_Tier"]
df_model.drop(columns=drop_eng, inplace=True)

FEATURES = [c for c in df_model.columns if c != "Exited"]
TARGET   = "Exited"

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n  Train : {X_train.shape[0]:,} rows | Test : {X_test.shape[0]:,} rows")
print(f"  Features ({len(FEATURES)}): {FEATURES}")
print(f"  Train churn rate: {y_train.mean():.1%} | Test churn rate: {y_test.mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PREDICTIVE MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 7: PREDICTIVE MODELLING")
print("="*70)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}

results = {}

for name, model in models.items():
    use_scaled = (name == "Logistic Regression")
    Xtr = X_train_sc if use_scaled else X_train
    Xte = X_test_sc  if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1":        f1_score(y_test, y_pred),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    # Cross-validation AUC
    cv_scores = cross_val_score(
        model, Xtr, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc", n_jobs=-1
    )
    results[name]["CV_AUC_Mean"] = cv_scores.mean()
    results[name]["CV_AUC_Std"]  = cv_scores.std()

    print(f"\n  {name}")
    print(f"    Accuracy : {results[name]['Accuracy']:.3f}")
    print(f"    Precision: {results[name]['Precision']:.3f}")
    print(f"    Recall   : {results[name]['Recall']:.3f}")
    print(f"    F1       : {results[name]['F1']:.3f}")
    print(f"    ROC-AUC  : {results[name]['ROC-AUC']:.3f}")
    print(f"    CV AUC   : {results[name]['CV_AUC_Mean']:.3f} ± {results[name]['CV_AUC_Std']:.3f}")


# ── FIG 5: Model Comparison Bar Chart ─────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in res.items() if k in ["Accuracy","Precision","Recall","F1","ROC-AUC"]}
    for name, res in results.items()
}).T

fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(metrics_df.columns))
width = 0.25
for i, (name, row) in enumerate(metrics_df.iterrows()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, row.values, width, label=name,
                  color=PALETTE[i], edgecolor="white", linewidth=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                f"{b.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics_df.columns, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison — NorthStar Churn Models")
ax.legend(loc="lower right")
ax.axhline(0.8, color=BRAND_GREY, linestyle=":", alpha=0.5, label="0.80 threshold")
save_fig("05_model_comparison")

# ── FIG 6: ROC Curves ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, lw=2.5, color=PALETTE[i], label=f"{name} (AUC = {res['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.500)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — NorthStar Churn Models")
ax.legend(loc="lower right")
ax.fill_between([0, 1], [0, 1], alpha=0.03, color=BRAND_GREY)
save_fig("06_roc_curves")

# ── FIG 7: Confusion Matrices ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", color=BRAND_BLUE)
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
save_fig("07_confusion_matrices")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FEATURE IMPORTANCE & BUSINESS INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 8: FEATURE IMPORTANCE")
print("="*70)

# Best model = Random Forest (by ROC-AUC)
best_name  = max(results, key=lambda n: results[n]["ROC-AUC"])
best_model = results[best_name]["model"]
print(f"\n  Best model: {best_name} (ROC-AUC = {results[best_name]['ROC-AUC']:.3f})")

# RF feature importance
rf_model = results["Random Forest"]["model"]
feat_imp  = pd.DataFrame({
    "Feature":   FEATURES,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(f"\n  Top 10 features by importance:\n{feat_imp.head(10).to_string(index=False)}")

# ── FIG 8: Feature Importance ─────────────────────────────────────────────────
top_n   = 12
top_imp = feat_imp.head(top_n)

fig, ax = plt.subplots(figsize=(11, 7))
colors_imp = [BRAND_RED if i < 3 else BRAND_BLUE if i < 7 else BRAND_TEAL
              for i in range(top_n)]
bars = ax.barh(top_imp["Feature"][::-1], top_imp["Importance"][::-1],
               color=colors_imp[::-1], edgecolor="white")
for bar, val in zip(bars, top_imp["Importance"][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 12 Churn Predictors — Random Forest Feature Importance")
ax.axvline(0, color="black", linewidth=0.5)

# Add business labels
business_labels = {
    "Age":                 "← Key demographic risk driver",
    "IsActiveMember":      "← Strongest behavioural signal",
    "Balance":             "← Financial engagement proxy",
    "NumOfProducts":       "← Product portfolio concentration risk",
    "Engagement_Score":    "← Composite engagement metric",
}
for bar, feat in zip(bars, top_imp["Feature"][::-1]):
    if feat in business_labels:
        ax.text(bar.get_width() + 0.025, bar.get_y() + bar.get_height()/2,
                business_labels[feat], va="center", fontsize=7.5,
                color=BRAND_GREY, style="italic")
save_fig("08_feature_importance")


# ── FIG 9: Risk Score Distribution ───────────────────────────────────────────
best_proba = results[best_name]["y_proba"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Churn Risk Score Distribution — NorthStar Portfolio", fontsize=14,
             fontweight="bold", color=BRAND_BLUE)

# Score distribution by outcome
for val, label, color in [(0,"Retained",BRAND_TEAL),(1,"Churned",BRAND_RED)]:
    mask = y_test == val
    axes[0].hist(best_proba[mask], bins=40, alpha=0.65, density=True,
                 color=color, label=label, edgecolor="white")
axes[0].axvline(0.5, color="black", linestyle="--", label="Default threshold (0.5)")
axes[0].set_xlabel("Predicted Churn Probability")
axes[0].set_ylabel("Density")
axes[0].set_title("Score Distribution by Actual Outcome")
axes[0].legend()

# Risk tiering
risk_tiers = pd.cut(best_proba,
                    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    labels=["Very Low\n(0–20%)", "Low\n(20–40%)",
                            "Medium\n(40–60%)", "High\n(60–80%)", "Critical\n(80–100%)"])
tier_counts = risk_tiers.value_counts().sort_index()
tier_colors = [BRAND_TEAL, "#77C4A0", BRAND_AMBER, BRAND_RED, "#8B0000"]
axes[1].bar(tier_counts.index, tier_counts.values, color=tier_colors, edgecolor="white")
for i, (idx, val) in enumerate(tier_counts.items()):
    axes[1].text(i, val + 5, f"{val:,}", ha="center", fontsize=9, fontweight="bold")
axes[1].set_title("Test Portfolio Risk Tier Distribution")
axes[1].set_ylabel("Number of Customers")
save_fig("09_risk_score_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FINANCIAL IMPACT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 9: FINANCIAL IMPACT ESTIMATION")
print("="*70)

# ── Assumptions (documented, challengeable) ───────────────────────────────────
TOTAL_CUSTOMERS      = 1_200_000   # NorthStar assumed book size
AVG_ANNUAL_REVENUE   = 850         # £ net revenue per customer (NIM + fees)
CHURN_RATE           = churn_rate
RETENTION_COST       = 85          # £ per outreach (offer + ops)
INTERVENTION_SUCCESS = 0.32        # 32% retention lift on targeted customers
MODEL_PRECISION      = results[best_name]["Precision"]
MODEL_RECALL         = results[best_name]["Recall"]

# Customers at risk
customers_at_risk = int(TOTAL_CUSTOMERS * CHURN_RATE)
revenue_at_risk   = customers_at_risk * AVG_ANNUAL_REVENUE

# Targetable churners (model identifies)
identifiable      = int(customers_at_risk * MODEL_RECALL)
false_positives   = int(identifiable / MODEL_PRECISION * (1 - MODEL_PRECISION))

# Revenue saved
retained_customers = int(identifiable * INTERVENTION_SUCCESS)
revenue_saved      = retained_customers * AVG_ANNUAL_REVENUE

# Intervention cost
total_interventions   = identifiable + false_positives
intervention_total    = total_interventions * RETENTION_COST

# Net benefit
net_benefit = revenue_saved - intervention_total
roi         = (net_benefit / intervention_total * 100) if intervention_total > 0 else 0

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  FINANCIAL IMPACT MODEL — KEY ASSUMPTIONS                       │
  ├─────────────────────────────────────────────────────────────────┤
  │  Total customer book         : {TOTAL_CUSTOMERS:>12,.0f}                   │
  │  Net revenue per customer    : £{AVG_ANNUAL_REVENUE:>11,.0f}/yr              │
  │  Portfolio churn rate        : {CHURN_RATE:>11.1%}                   │
  │  Retention cost per customer : £{RETENTION_COST:>11,.0f}                   │
  │  Intervention success rate   : {INTERVENTION_SUCCESS:>11.0%}                   │
  │  Model precision             : {MODEL_PRECISION:>11.1%}                   │
  │  Model recall                : {MODEL_RECALL:>11.1%}                   │
  ├─────────────────────────────────────────────────────────────────┤
  │  RESULTS                                                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  Customers at risk           : {customers_at_risk:>12,.0f}                   │
  │  Total revenue at risk       : £{revenue_at_risk:>10,.0f}                   │
  │  Model-identified churners   : {identifiable:>12,.0f}                   │
  │  False positives (wasted)    : {false_positives:>12,.0f}                   │
  │  Customers successfully ret. : {retained_customers:>12,.0f}                   │
  │  Revenue saved (annual)      : £{revenue_saved:>10,.0f}                   │
  │  Total intervention cost     : £{intervention_total:>10,.0f}                   │
  │  NET ANNUAL BENEFIT          : £{net_benefit:>10,.0f}                   │
  │  ROI on retention spend      : {roi:>11.0f}%                   │
  └─────────────────────────────────────────────────────────────────┘
""")

# ── FIG 10: Financial Waterfall ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
categories = [
    "Revenue\nAt Risk",
    "Model-Identified\nRevenue at Risk",
    "Revenue\nSaved",
    "Intervention\nCost",
    "Net Annual\nBenefit"
]
values = [revenue_at_risk, identifiable * AVG_ANNUAL_REVENUE,
          revenue_saved, -intervention_total, net_benefit]
bar_colors = [BRAND_RED, BRAND_AMBER, BRAND_TEAL, BRAND_GREY, BRAND_BLUE]

bars = ax.bar(categories, [abs(v)/1e6 for v in values], color=bar_colors,
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, values):
    sign = "+" if val >= 0 else "-"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{sign}£{abs(val)/1e6:.1f}M", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("£ Millions")
ax.set_title("Financial Impact Waterfall — NorthStar Churn Model (Annual Estimate)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}M"))
for bar in bars[-1:]:
    bar.set_edgecolor(BRAND_BLUE)
    bar.set_linewidth(2.5)
save_fig("10_financial_impact_waterfall")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CHURN RISK SEGMENTATION (FINAL OUTPUT)
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("  SECTION 10: RISK SEGMENTATION & ACTION MATRIX")
print("="*70)

test_output = X_test.copy()
test_output["Actual_Churn"]   = y_test.values
test_output["Churn_Prob"]     = best_proba
test_output["Risk_Tier"]      = pd.cut(
    best_proba, bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["Low", "Medium", "High", "Critical"]
)

risk_summary = test_output.groupby("Risk_Tier", observed=True).agg(
    Customers    = ("Churn_Prob", "count"),
    Avg_Churn_Prob = ("Churn_Prob", "mean"),
    Actual_Churned = ("Actual_Churn", "sum"),
).reset_index()
risk_summary["Churn_Rate"]     = (risk_summary["Actual_Churned"] / risk_summary["Customers"] * 100).round(1)
risk_summary["Est_Annual_Rev"] = (risk_summary["Customers"] * (TOTAL_CUSTOMERS/len(df)) * AVG_ANNUAL_REVENUE / 1e6).round(1)

print(f"\n  Risk Tier Action Matrix:\n")
print(f"  {'Tier':<12} {'Customers':>10} {'Avg Prob':>10} {'Actual Churn%':>14} {'Est Rev (£M)':>14}")
print(f"  {'─'*60}")
for _, row in risk_summary.iterrows():
    print(f"  {row['Risk_Tier']:<12} {row['Customers']:>10,} {row['Avg_Churn_Prob']:>10.1%} "
          f"{row['Churn_Rate']:>13.1f}% {row['Est_Annual_Rev']:>13.1f}M")

print(f"""
  ─────────────────────────────────────────────────────────────────────
  ACTION RECOMMENDATIONS:
    CRITICAL (>70% prob): Immediate personalised retention call
    HIGH     (50–70%):    Targeted digital offer + product review
    MEDIUM   (30–50%):    Automated re-engagement campaign
    LOW      (<30%):      Monitor; no active intervention required
  ─────────────────────────────────────────────────────────────────────
""")

# ── FIG 11: Risk Tier Heatmap ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Customer Risk Segmentation — NorthStar Portfolio", fontsize=14,
             fontweight="bold", color=BRAND_BLUE)

# Customer count by tier
tier_colors_bar = [BRAND_TEAL, BRAND_AMBER, BRAND_RED, "#8B0000"]
axes[0].bar(risk_summary["Risk_Tier"], risk_summary["Customers"],
            color=tier_colors_bar, edgecolor="white")
axes[0].set_title("Customers by Risk Tier")
axes[0].set_ylabel("No. Customers (Test Set)")
for i, (_, row) in enumerate(risk_summary.iterrows()):
    axes[0].text(i, row["Customers"] + 3, f"{row['Customers']:,}",
                 ha="center", fontsize=10, fontweight="bold")

# Revenue at risk by tier (extrapolated)
rev_at_risk = risk_summary["Customers"] * (TOTAL_CUSTOMERS/len(df)) * AVG_ANNUAL_REVENUE * risk_summary["Churn_Rate"]/100 / 1e6
axes[1].bar(risk_summary["Risk_Tier"], rev_at_risk, color=tier_colors_bar, edgecolor="white")
axes[1].set_title("Estimated Annual Revenue at Risk by Tier (£M)")
axes[1].set_ylabel("Revenue at Risk (£M)")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}M"))
for i, val in enumerate(rev_at_risk):
    axes[1].text(i, val + 0.1, f"£{val:.1f}M", ha="center", fontsize=10, fontweight="bold")
save_fig("11_risk_segmentation")

print("\n" + "="*70)
print("  ANALYSIS COMPLETE")
print(f"  All visualisations saved to: ./{OUTPUT_DIR}/")
print("="*70 + "\n")
