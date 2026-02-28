"""
================================================================================
NorthStar Retail Bank — Customer Churn Prediction & Retention Value Analysis
Streamlit Dashboard — Consultancy Edition
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NorthStar Bank — Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── BRAND COLOURS ─────────────────────────────────────────────────────────────
BRAND_BLUE  = "#003865"
BRAND_TEAL  = "#00A499"
BRAND_AMBER = "#F0A500"
BRAND_RED   = "#C0392B"
BRAND_GREY  = "#6C757D"
PALETTE     = [BRAND_BLUE, BRAND_TEAL, BRAND_AMBER, BRAND_RED, BRAND_GREY]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "font.family":      "DejaVu Sans",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "legend.frameon":   False,
})

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700;800&family=Barlow+Condensed:wght@700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Barlow', sans-serif; }

  /* Header bar */
  .ns-header {
    background: linear-gradient(135deg, #003865 0%, #005a9c 100%);
    padding: 2rem 2.5rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .ns-header h1 {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: 0.5px;
  }
  .ns-header .subtitle {
    color: #7ec8e3;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 0.25rem;
    letter-spacing: 1px;
    text-transform: uppercase;
  }
  .ns-badge {
    background: #00A499;
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Metric cards */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  .metric-card {
    flex: 1;
    background: white;
    border: 1px solid #e8ecf0;
    border-left: 4px solid #003865;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .metric-card.teal  { border-left-color: #00A499; }
  .metric-card.amber { border-left-color: #F0A500; }
  .metric-card.red   { border-left-color: #C0392B; }
  .metric-card .label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6C757D;
    margin-bottom: 0.3rem;
  }
  .metric-card .value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #003865;
    line-height: 1;
  }
  .metric-card .delta {
    font-size: 0.78rem;
    color: #6C757D;
    margin-top: 0.2rem;
  }

  /* Section headers */
  .section-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #003865;
    border-bottom: 2px solid #003865;
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  /* Insight callout */
  .insight-box {
    background: #f0f7ff;
    border-left: 4px solid #003865;
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #1a2b3c;
  }
  .insight-box strong { color: #003865; }

  /* Action matrix table */
  .action-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  .action-table th {
    background: #003865; color: white; padding: 0.6rem 1rem;
    text-align: left; font-weight: 700; letter-spacing: 0.5px;
  }
  .action-table td { padding: 0.55rem 1rem; border-bottom: 1px solid #e8ecf0; }
  .action-table tr:nth-child(even) td { background: #f8f9fa; }
  .pill {
    display: inline-block; padding: 0.2rem 0.7rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
  }
  .pill-critical { background:#8B0000; color:white; }
  .pill-high     { background:#C0392B; color:white; }
  .pill-medium   { background:#F0A500; color:#1a1a1a; }
  .pill-low      { background:#00A499; color:white; }

  div[data-testid="stSidebar"] { background: #f4f6f9; }
  div[data-testid="stSidebar"] .stMarkdown h2 {
    color: #003865; font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800; text-transform: uppercase; letter-spacing: 1px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Building dataset…")
def load_data():
    try:
        df_raw = pd.read_csv("Churn_Modelling.csv")
    except FileNotFoundError:
        rng = np.random.default_rng(42)
        n = 10_000
        geography = rng.choice(["France", "Germany", "Spain"], n, p=[0.50, 0.25, 0.25])
        gender    = rng.choice(["Male", "Female"], n, p=[0.545, 0.455])
        age       = rng.integers(18, 92, n)
        tenure    = rng.integers(0, 11, n)
        num_prod  = rng.choice([1, 2, 3, 4], n, p=[0.505, 0.459, 0.026, 0.010])
        has_card  = rng.integers(0, 2, n)
        is_active = rng.integers(0, 2, n)
        credit    = np.clip(rng.normal(650, 97, n).astype(int), 350, 850)
        balance   = np.where(rng.random(n) < 0.36, 0, np.clip(rng.normal(76_485, 62_397, n), 0, 250_000))
        salary    = np.clip(rng.normal(100_090, 57_510, n), 11_500, 199_999)
        logit = (-2.0 + 0.035*(age-40) + 0.6*(geography=="Germany").astype(int)
                 - 0.4*is_active + 0.5*(num_prod>=3).astype(int)
                 - 0.3*(balance>0).astype(int) - 0.02*(credit-650)/97)
        prob_churn = 1/(1+np.exp(-logit))
        exited = rng.binomial(1, prob_churn)
        df_raw = pd.DataFrame({
            "RowNumber":range(1,n+1),"CustomerId":rng.integers(15_000_000,16_000_000,n),
            "Surname":["Client_"+str(i) for i in range(n)],"CreditScore":credit,
            "Geography":geography,"Gender":gender,"Age":age,"Tenure":tenure,
            "Balance":balance.round(2),"NumOfProducts":num_prod,"HasCrCard":has_card,
            "IsActiveMember":is_active,"EstimatedSalary":salary.round(2),"Exited":exited,
        })
    return df_raw


@st.cache_data(show_spinner="Engineering features…")
def engineer_features(df_raw):
    df = df_raw.copy()
    df.drop(columns=["RowNumber","CustomerId","Surname"], errors="ignore", inplace=True)
    df.drop_duplicates(inplace=True)
    df["CreditScore"] = df["CreditScore"].clip(lower=300)
    df["Balance_Salary_Ratio"] = np.where(df["EstimatedSalary"]>0, df["Balance"]/df["EstimatedSalary"], 0)
    df["Age_Band"]    = pd.cut(df["Age"], bins=[0,30,40,50,60,100], labels=["<30","30–39","40–49","50–59","60+"])
    df["Tenure_Band"] = pd.cut(df["Tenure"], bins=[-1,1,3,6,10], labels=["0–1 yr","2–3 yr","4–6 yr","7+ yr"])
    df["Zero_Balance_Flag"] = (df["Balance"]==0).astype(int)
    df["Engagement_Score"]  = df["NumOfProducts"]+df["HasCrCard"]+df["IsActiveMember"]
    df["High_Value_Flag"]   = (df["Balance"]>100_000).astype(int)
    df["Credit_Tier"] = pd.cut(df["CreditScore"], bins=[0,579,669,739,799,900],
                               labels=["Poor","Fair","Good","Very Good","Excellent"])
    return df


@st.cache_resource(show_spinner="Training models…")
def train_models(df):
    df_model = df.copy()
    le = LabelEncoder()
    for col in ["Geography","Gender"]:
        df_model[col] = le.fit_transform(df_model[col])
    df_model.drop(columns=["Age_Band","Tenure_Band","Credit_Tier"], inplace=True)
    FEATURES = [c for c in df_model.columns if c != "Exited"]
    X = df_model[FEATURES]; y = df_model["Exited"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train); X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200,max_depth=8,class_weight="balanced",random_state=42,n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=42),
    }
    results = {}
    for name, model in models.items():
        Xtr = X_train_sc if name=="Logistic Regression" else X_train
        Xte = X_test_sc  if name=="Logistic Regression" else X_test
        model.fit(Xtr,y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:,1]
        cv = cross_val_score(model,Xtr,y_train,cv=StratifiedKFold(5,shuffle=True,random_state=42),scoring="roc_auc",n_jobs=-1)
        results[name] = {
            "model":model,"y_pred":y_pred,"y_proba":y_proba,
            "Accuracy":accuracy_score(y_test,y_pred),"Precision":precision_score(y_test,y_pred),
            "Recall":recall_score(y_test,y_pred),"F1":f1_score(y_test,y_pred),
            "ROC-AUC":roc_auc_score(y_test,y_proba),
            "CV_AUC_Mean":cv.mean(),"CV_AUC_Std":cv.std(),
        }
    best_name = max(results,key=lambda n:results[n]["ROC-AUC"])
    feat_imp  = pd.DataFrame({"Feature":FEATURES,"Importance":results["Random Forest"]["model"].feature_importances_}).sort_values("Importance",ascending=False)
    return results, best_name, feat_imp, X_test, y_test, FEATURES


# ── Load everything ────────────────────────────────────────────────────────────
df_raw = load_data()
df     = engineer_features(df_raw)
results, best_name, feat_imp, X_test, y_test, FEATURES = train_models(df)
churn_rate = df["Exited"].mean()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", ["Executive Summary","EDA & Segmentation","Model Performance","Financial Impact","Risk Action Matrix"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("## Financial Assumptions")
    TOTAL_CUSTOMERS    = st.number_input("Total Customer Book", value=1_200_000, step=50_000)
    AVG_ANNUAL_REVENUE = st.number_input("Net Revenue per Customer (£/yr)", value=850, step=50)
    RETENTION_COST     = st.number_input("Retention Cost per Customer (£)", value=85, step=5)
    INTERVENTION_SUCCESS = st.slider("Intervention Success Rate", 0.10, 0.60, 0.32, 0.01, format="%.0%%")
    st.markdown("---")
    st.caption("NorthStar Retail Bank · Q1 2025\nChurn Intelligence Platform · v1.0")


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ns-header">
  <div>
    <div class="h1">🏦 NorthStar Retail Bank</div>
    <h1>Customer Churn Intelligence</h1>
    <div class="subtitle">Predictive Analytics · Retention Strategy · Revenue Recovery</div>
  </div>
  <div>
    <div class="ns-badge">Confidential</div><br><br>
    <div class="ns-badge" style="background:#F0A500;color:#1a1a1a;">Q1 2025</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    # KPI cards
    churn_n = df["Exited"].sum()
    rev_risk = int(TOTAL_CUSTOMERS * churn_rate * AVG_ANNUAL_REVENUE)
    best_auc = results[best_name]["ROC-AUC"]

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="label">Portfolio Churn Rate</div>
        <div class="value">{churn_rate:.1%}</div>
        <div class="delta">{churn_n:,} customers at risk of leaving</div>
      </div>
      <div class="metric-card teal">
        <div class="label">Revenue at Risk (Annual)</div>
        <div class="value">£{rev_risk/1e6:.1f}M</div>
        <div class="delta">Based on £{AVG_ANNUAL_REVENUE:,} net revenue/customer</div>
      </div>
      <div class="metric-card amber">
        <div class="label">Best Model ROC-AUC</div>
        <div class="value">{best_auc:.3f}</div>
        <div class="delta">{best_name}</div>
      </div>
      <div class="metric-card red">
        <div class="label">Total Customers</div>
        <div class="value">{TOTAL_CUSTOMERS/1e6:.1f}M</div>
        <div class="delta">Extrapolated book size</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)

    churn_geo  = df.groupby("Geography")["Exited"].mean()
    churn_age  = df.groupby("Age_Band")["Exited"].mean()
    churn_act  = df.groupby("IsActiveMember")["Exited"].mean()
    churn_prod = df.groupby("NumOfProducts")["Exited"].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="insight-box">
          <strong>🌍 Geography Risk:</strong> German customers churn at
          <strong>{churn_geo.get('Germany',0):.1%}</strong> vs France at
          <strong>{churn_geo.get('France',0):.1%}</strong> —
          a <strong>{(churn_geo.get('Germany',0)-churn_geo.get('France',0))*100:.0f}pp gap</strong>
          requiring market-specific retention strategy.
        </div>
        <div class="insight-box">
          <strong>📅 Age Concentration:</strong> Customers aged 40–59 show peak churn
          ({churn_age.get('40–49',0):.1%} / {churn_age.get('50–59',0):.1%}).
          This cohort typically holds the highest AUM — disproportionate revenue exposure.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="insight-box">
          <strong>⚡ Inactivity Signal:</strong> Inactive members churn at
          <strong>{churn_act.get(0,0):.1%}</strong> vs
          <strong>{churn_act.get(1,0):.1%}</strong> for active —
          a <strong>{(churn_act.get(0,0)-churn_act.get(1,0))*100:.0f}pp difference</strong>.
          Re-engagement is the single highest-leverage intervention.
        </div>
        <div class="insight-box">
          <strong>📦 Product Bundling Risk:</strong> 3–4 product holders churn at
          >{churn_prod.get(3,0):.0%} — suggesting forced cross-sell or poor product-market fit
          rather than genuine multi-product engagement.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Churn Overview</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        retain_n = len(df) - churn_n
        wedges, texts, autotexts = ax.pie(
            [retain_n, churn_n], labels=["Retained","Churned"],
            colors=[BRAND_TEAL, BRAND_RED], autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2})
        for at in autotexts: at.set_fontweight("bold")
        ax.set_title("Portfolio Churn Split")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        churn_by_country = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
        bars = ax.bar(churn_by_country.index, churn_by_country.values*100,
                      color=[BRAND_RED, BRAND_AMBER, BRAND_TEAL])
        ax.set_title("Churn Rate by Geography"); ax.set_ylabel("Churn Rate (%)")
        ax.axhline(churn_rate*100, color=BRAND_GREY, linestyle="--", linewidth=1.5, label="Portfolio Avg")
        ax.legend()
        for bar, val in zip(bars, churn_by_country.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA & Segmentation":
    st.markdown('<div class="section-header">Churn by Segment</div>', unsafe_allow_html=True)

    def churn_bar(ax, col, title, xticklabels=None, palette=None):
        data = df.groupby(col)["Exited"].mean().reset_index()
        data.columns = [col,"ChurnRate"]
        pal = palette or [BRAND_BLUE]*len(data)
        bars = ax.bar(data[col].astype(str), data["ChurnRate"]*100, color=pal, edgecolor="white")
        ax.axhline(churn_rate*100, color=BRAND_GREY, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_title(title); ax.set_ylabel("Churn Rate (%)")
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f}%", ha="center", fontsize=8)
        if xticklabels: ax.set_xticklabels(xticklabels)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Churn Rate by Customer Segment", fontsize=14, fontweight="bold", color=BRAND_BLUE)
    churn_bar(axes[0,0], "Age_Band", "Churn by Age Band", palette=[BRAND_TEAL,BRAND_BLUE,BRAND_AMBER,BRAND_RED,"#8B0000"])
    churn_bar(axes[0,1], "NumOfProducts", "Churn by Number of Products", palette=[BRAND_TEAL,BRAND_BLUE,BRAND_RED,"#8B0000"])
    churn_bar(axes[1,0], "IsActiveMember", "Churn by Activity Status", xticklabels=["Inactive","Active"], palette=[BRAND_RED,BRAND_TEAL])
    churn_bar(axes[1,1], "Gender", "Churn by Gender", palette=[BRAND_BLUE,BRAND_AMBER])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">Feature Distributions: Churned vs Retained</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, xlabel in zip(axes,["Age","CreditScore","Balance"],["Age (years)","Credit Score","Account Balance (£)"]):
        for val, label, color in [(0,"Retained",BRAND_TEAL),(1,"Churned",BRAND_RED)]:
            ax.hist(df[df["Exited"]==val][col], bins=30, alpha=0.65, label=label, color=color, density=True, edgecolor="white")
        ax.set_xlabel(xlabel); ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
    numeric_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard",
                    "IsActiveMember","EstimatedSalary","Balance_Salary_Ratio",
                    "Zero_Balance_Flag","Engagement_Score","High_Value_Flag","Exited"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.5, linecolor="white", annot_kws={"size":8}, ax=ax)
    ax.set_title("Correlation Matrix — NorthStar Customer Features", pad=15)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="section-header">Hypothesis Testing</div>', unsafe_allow_html=True)
    churned_age  = df[df["Exited"]==1]["Age"]
    retained_age = df[df["Exited"]==0]["Age"]
    t_stat, p_val = stats.ttest_ind(churned_age, retained_age)
    contingency   = pd.crosstab(df["Geography"], df["Exited"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
    corr_active, p_active = stats.pointbiserialr(df["IsActiveMember"], df["Exited"])

    col1, col2, col3 = st.columns(3)
    for col_widget, title, stat_label, stat_val, p, result in [
        (col1, "H1: Churned customers are older", "t-statistic", t_stat, p_val,
         f"Churned mean: {churned_age.mean():.1f} yrs | Retained: {retained_age.mean():.1f} yrs"),
        (col2, "H2: Churn varies by geography", "Chi-squared", chi2, p_chi,
         f"χ² = {chi2:.2f}, DoF = {dof}"),
        (col3, "H3: Active membership reduces churn", "Point-biserial r", corr_active, p_active,
         f"r = {corr_active:.3f}"),
    ]:
        verdict = "✅ REJECT H₀" if p < 0.05 else "❌ FAIL TO REJECT H₀"
        color   = "#00A499" if p < 0.05 else "#C0392B"
        col_widget.markdown(f"""
        <div class="insight-box">
          <strong>{title}</strong><br>
          {result}<br>
          p-value: <strong>{p:.2e}</strong><br>
          <span style="color:{color};font-weight:700">{verdict}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown('<div class="section-header">Model Scorecard</div>', unsafe_allow_html=True)

    metrics_data = {
        name: {k: v for k,v in res.items() if k in ["Accuracy","Precision","Recall","F1","ROC-AUC","CV_AUC_Mean"]}
        for name, res in results.items()
    }
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_df.columns = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","CV AUC (mean)"]
    st.dataframe(
        metrics_df.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0),
        use_container_width=True
    )

    st.markdown('<div class="section-header">ROC Curves</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
            ax.plot(fpr, tpr, lw=2.5, color=PALETTE[i], label=f"{name} (AUC={res['ROC-AUC']:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1.5,label="Random (0.500)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models"); ax.legend(loc="lower right")
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (name, res) in zip(axes, results.items()):
            cm = confusion_matrix(y_test, res["y_pred"])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained","Churned"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(name, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    top_n  = 12
    top_imp = feat_imp.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = [BRAND_RED if i<3 else BRAND_BLUE if i<7 else BRAND_TEAL for i in range(top_n)]
    ax.barh(top_imp["Feature"][::-1], top_imp["Importance"][::-1], color=colors_imp[::-1], edgecolor="white")
    for bar, val in zip(ax.patches, top_imp["Importance"][::-1]):
        ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Top 12 Churn Predictors — Random Forest")
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
    best_proba = results[best_name]["y_proba"]
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for val, label, color in [(0,"Retained",BRAND_TEAL),(1,"Churned",BRAND_RED)]:
            mask = y_test == val
            ax.hist(best_proba[mask], bins=40, alpha=0.65, density=True, color=color, label=label, edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", label="Threshold (0.5)")
        ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Density")
        ax.set_title("Score Distribution by Actual Outcome"); ax.legend()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        risk_tiers = pd.cut(best_proba, bins=[0,0.2,0.4,0.6,0.8,1.0],
                            labels=["Very Low","Low","Medium","High","Critical"])
        tier_counts = risk_tiers.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(tier_counts.index, tier_counts.values,
               color=[BRAND_TEAL,"#77C4A0",BRAND_AMBER,BRAND_RED,"#8B0000"], edgecolor="white")
        for i, (idx, val) in enumerate(tier_counts.items()):
            ax.text(i, val+5, f"{val:,}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Test Portfolio Risk Tier Distribution"); ax.set_ylabel("Customers")
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINANCIAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Financial Impact":
    MODEL_PRECISION = results[best_name]["Precision"]
    MODEL_RECALL    = results[best_name]["Recall"]

    customers_at_risk  = int(TOTAL_CUSTOMERS * churn_rate)
    revenue_at_risk    = customers_at_risk * AVG_ANNUAL_REVENUE
    identifiable       = int(customers_at_risk * MODEL_RECALL)
    false_positives    = int(identifiable / MODEL_PRECISION * (1 - MODEL_PRECISION)) if MODEL_PRECISION > 0 else 0
    retained_customers = int(identifiable * INTERVENTION_SUCCESS)
    revenue_saved      = retained_customers * AVG_ANNUAL_REVENUE
    total_interventions= identifiable + false_positives
    intervention_total = total_interventions * RETENTION_COST
    net_benefit        = revenue_saved - intervention_total
    roi                = (net_benefit / intervention_total * 100) if intervention_total > 0 else 0

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card red">
        <div class="label">Revenue at Risk</div>
        <div class="value">£{revenue_at_risk/1e6:.1f}M</div>
        <div class="delta">{customers_at_risk:,} customers likely to churn</div>
      </div>
      <div class="metric-card teal">
        <div class="label">Revenue Saved</div>
        <div class="value">£{revenue_saved/1e6:.1f}M</div>
        <div class="delta">{retained_customers:,} customers retained via model</div>
      </div>
      <div class="metric-card amber">
        <div class="label">Intervention Cost</div>
        <div class="value">£{intervention_total/1e6:.1f}M</div>
        <div class="delta">{total_interventions:,} total outreaches at £{RETENTION_COST}/customer</div>
      </div>
      <div class="metric-card">
        <div class="label">Net Annual Benefit</div>
        <div class="value">£{net_benefit/1e6:.1f}M</div>
        <div class="delta">ROI: {roi:.0f}% on retention spend</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Financial Waterfall</div>', unsafe_allow_html=True)
    categories = ["Revenue\nat Risk","Model-Identified\nRevenue","Revenue\nSaved","Intervention\nCost","Net Annual\nBenefit"]
    values     = [revenue_at_risk, identifiable*AVG_ANNUAL_REVENUE, revenue_saved, -intervention_total, net_benefit]
    bar_colors = [BRAND_RED, BRAND_AMBER, BRAND_TEAL, BRAND_GREY, BRAND_BLUE]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(categories, [abs(v)/1e6 for v in values], color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        sign = "+" if val >= 0 else "-"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f"{sign}£{abs(val)/1e6:.1f}M", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("£ Millions")
    ax.set_title("Financial Impact Waterfall — NorthStar Churn Model (Annual Estimate)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}M"))
    bars[-1].set_edgecolor(BRAND_BLUE); bars[-1].set_linewidth(2.5)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Model Assumptions</div>', unsafe_allow_html=True)
    assumptions = pd.DataFrame({
        "Parameter": ["Total Customer Book","Net Revenue per Customer","Portfolio Churn Rate",
                      "Retention Cost per Customer","Intervention Success Rate",
                      "Model Precision","Model Recall"],
        "Value": [f"{TOTAL_CUSTOMERS:,}", f"£{AVG_ANNUAL_REVENUE:,}/yr", f"{churn_rate:.1%}",
                  f"£{RETENTION_COST:,}", f"{INTERVENTION_SUCCESS:.0%}",
                  f"{MODEL_PRECISION:.1%}", f"{MODEL_RECALL:.1%}"],
        "Notes": ["Assumed NorthStar book size","Net Interest Margin + fees proxy",
                  "Empirical — dataset derived","Outreach + offer + ops estimate",
                  "Lift from targeted personalised offer","Best model on test set","Best model on test set"],
    })
    st.dataframe(assumptions, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RISK ACTION MATRIX
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Risk Action Matrix":
    best_proba = results[best_name]["y_proba"]
    test_output = X_test.copy()
    test_output["Actual_Churn"] = y_test.values
    test_output["Churn_Prob"]   = best_proba
    test_output["Risk_Tier"]    = pd.cut(best_proba, bins=[0,0.3,0.5,0.7,1.0], labels=["Low","Medium","High","Critical"])

    risk_summary = test_output.groupby("Risk_Tier", observed=True).agg(
        Customers=("Churn_Prob","count"),
        Avg_Churn_Prob=("Churn_Prob","mean"),
        Actual_Churned=("Actual_Churn","sum"),
    ).reset_index()
    risk_summary["Churn_Rate_%"] = (risk_summary["Actual_Churned"]/risk_summary["Customers"]*100).round(1)
    risk_summary["Est_Rev_at_Risk_£M"] = (risk_summary["Customers"]*(TOTAL_CUSTOMERS/len(df))*AVG_ANNUAL_REVENUE*risk_summary["Churn_Rate_%"]/100/1e6).round(1)

    st.markdown('<div class="section-header">Risk Tier Action Matrix</div>', unsafe_allow_html=True)

    actions = {
        "Critical": ("🔴", "pill-critical", "Immediate personalised retention call within 24h. Assign dedicated relationship manager. Offer tailored product upgrade or loyalty reward."),
        "High":     ("🟠", "pill-high",     "Targeted digital offer within 72h. Product portfolio review. Flag to branch manager for proactive outreach."),
        "Medium":   ("🟡", "pill-medium",   "Automated re-engagement campaign (email + app push). Educational content on product benefits. 30-day monitoring period."),
        "Low":      ("🟢", "pill-low",      "No active intervention required. Standard NPS survey. Monitor for signal deterioration quarterly."),
    }

    table_rows = ""
    for _, row in risk_summary.iterrows():
        tier = str(row["Risk_Tier"])
        icon, pill_class, action_text = actions.get(tier, ("⚪", "pill-low", ""))
        table_rows += f"""
        <tr>
          <td><span class="pill {pill_class}">{icon} {tier}</span></td>
          <td>{row['Customers']:,}</td>
          <td>{row['Avg_Churn_Prob']:.1%}</td>
          <td>{row['Churn_Rate_%']:.1f}%</td>
          <td>£{row['Est_Rev_at_Risk_£M']:.1f}M</td>
          <td style="font-size:0.82rem;color:#444">{action_text}</td>
        </tr>"""

    st.markdown(f"""
    <table class="action-table">
      <thead>
        <tr>
          <th>Risk Tier</th><th>Customers</th><th>Avg Churn Prob</th>
          <th>Actual Churn %</th><th>Rev at Risk (£M)</th><th>Recommended Action</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Visualisation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    tier_colors = [BRAND_TEAL, BRAND_AMBER, BRAND_RED, "#8B0000"]

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(risk_summary["Risk_Tier"], risk_summary["Customers"], color=tier_colors, edgecolor="white")
        for i, (_, row) in enumerate(risk_summary.iterrows()):
            ax.text(i, row["Customers"]+3, f"{row['Customers']:,}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Customers by Risk Tier"); ax.set_ylabel("No. Customers (Test Set)")
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        rev_at_risk_vals = risk_summary["Customers"]*(TOTAL_CUSTOMERS/len(df))*AVG_ANNUAL_REVENUE*risk_summary["Churn_Rate_%"]/100/1e6
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(risk_summary["Risk_Tier"], rev_at_risk_vals, color=tier_colors, edgecolor="white")
        for i, val in enumerate(rev_at_risk_vals):
            ax.text(i, val+0.1, f"£{val:.1f}M", ha="center", fontsize=9, fontweight="bold")
        ax.set_title("Estimated Annual Revenue at Risk by Tier"); ax.set_ylabel("Revenue at Risk (£M)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"£{x:.0f}M"))
        st.pyplot(fig, use_container_width=True); plt.close()

    # Download scored customers
    st.markdown('<div class="section-header">Export Scored Customers</div>', unsafe_allow_html=True)
    csv = test_output[["Churn_Prob","Risk_Tier","Actual_Churn"]].to_csv(index=False)
    st.download_button("⬇️ Download Scored Customer List (CSV)", csv, "northstar_scored_customers.csv", "text/csv")