"""
Credit Risk Intelligence Dashboard
===================================
Pages: Overview | Live Prediction
Models: XGBoost, Random Forest, LightGBM, Gradient Boosting

Run: streamlit run credit_risk_dashboard.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings('ignore')

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  – dark navy theme, monospace numbers, teal accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main                       { background-color: #0E1117; }
.block-container            { padding: 1.5rem 2rem; }

/* ── KPI cards ─────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
    border: 1px solid #2d3561;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.2s;
}
.metric-card:hover  { transform: translateY(-2px); }
.metric-label       { color: #8892b0; font-size: 0.78rem; font-weight: 600;
                       letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value       { color: #e6f1ff; font-size: 1.9rem; font-weight: 700;
                       font-family: 'JetBrains Mono', monospace; }

/* ── Section headings ───────────────────────────────────────────────────── */
.section-header {
    color: #ccd6f6; font-size: 1.1rem; font-weight: 700;
    border-left: 4px solid #64ffda;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── Prediction result boxes ────────────────────────────────────────────── */
.pred-approved {
    background: linear-gradient(135deg, #0d4b3b, #0a3b2e);
    border: 1px solid #64ffda; border-radius: 16px;
    padding: 1.5rem; text-align: center;
    color: #64ffda; font-size: 1.4rem; font-weight: 700;
}
.pred-rejected {
    background: linear-gradient(135deg, #4b0d0d, #3b0a0a);
    border: 1px solid #ff6b6b; border-radius: 16px;
    padding: 1.5rem; text-align: center;
    color: #ff6b6b; font-size: 1.4rem; font-weight: 700;
}

/* ── Risk label colours ─────────────────────────────────────────────────── */
.risk-low  { color: #64ffda; font-weight: 700; }
.risk-mid  { color: #ffd700; font-weight: 700; }
.risk-high { color: #ff6b6b; font-weight: 700; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR  = "/Users/muditpanwar/PycharmProjects/credit_risk_analysis_project/output/models"      # folder where trained .pkl files live
TARGET_COL = "LoanApproved"

# Which .pkl files to look for  (CatBoost removed)
MODEL_FILES = {
    "xgboost_best.pkl":           "XGBoost",
    "random_forest_best.pkl":     "Random Forest",
    "lightgbm_best.pkl":          "LightGBM",
    "gradient_boosting_best.pkl": "Gradient Boosting",
}

# Ordinal encoding maps – must match what the notebook used during training
ORDINAL_MAPS = {
    "EducationLevel": {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4},
    "Credit_Tier":    {"1. Poor": 0, "2. Fair": 1, "3. Good": 2, "4. Very Good": 3, "5. Excellent": 4},
    "Risk_Band":      {"Low Risk": 0, "Medium Risk": 1, "High Risk": 2},
    "Income_Bracket": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
    "DTI_Bucket":     {"Healthy (<30%)": 0, "Moderate (30-40%)": 1, "High (40-50%)": 2, "Very High (>50%)": 3},
    "AgeGroup":       {"Young Adult": 0, "Adult": 1, "Middle-Aged": 2, "Senior": 3},
}

# Label-encoding maps for categorical columns (same as training)
LABEL_MAPS = {
    "EmploymentStatus":    {"Employed": 0, "Part-Time": 1, "Self-Employed": 2, "Unemployed": 3},
    "MaritalStatus":       {"Divorced": 0, "Married": 1, "Single": 2, "Widowed": 3},
    "HomeOwnershipStatus": {"Mortgage": 0, "Own": 1, "Rent": 2},
    "LoanPurpose":         {"Auto": 0, "Business": 1, "Debt Consolidation": 2,
                            "Education": 3, "Home": 4, "Medical": 5, "Personal": 6},
}

# Columns to remove before feeding data to a model (they cause data leakage)
LEAKAGE_COLS = [
    "ApplicationDate", "App_Month", "ApplicationStatus",
    "Repayment_Status", "InterestRate", "MonthlyLoanPayment",
]

# matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor":   "#161b27",
    "axes.edgecolor":   "#2d3561",
    "axes.labelcolor":  "#8892b0",
    "xtick.color":      "#8892b0",
    "ytick.color":      "#8892b0",
    "text.color":       "#ccd6f6",
    "grid.color":       "#21262d",
    "grid.alpha":        0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    """Read CSV or Excel upload into a DataFrame."""
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    """
    Load all available trained models from the /models folder.
    Returns a dict:  { display_name: model_object }
    """
    loaded = {}
    if not os.path.exists(MODEL_DIR):
        return loaded
    for fname, display_name in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            loaded[display_name] = joblib.load(path)
    return loaded


def preprocess_df(df: pd.DataFrame):
    """
    Clean the raw DataFrame so it can be used for EDA or model evaluation.
    - Drops leakage columns
    - Separates the target column (LoanApproved)
    - Encodes ordinal + categorical columns numerically
    Returns: (processed_df, target_series_or_None)
    """
    df = df.copy()

    # Remove columns that would leak the answer
    df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], inplace=True)

    # Separate target
    y = df.pop(TARGET_COL) if TARGET_COL in df.columns else None

    # Apply ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # Label-encode any remaining string/category columns
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df, y


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: HTML metric card
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str) -> str:
    """Return an HTML snippet for a styled KPI card."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk\n### Intelligence Platform")
    st.markdown("---")

    # Dataset upload
    uploaded_file = st.file_uploader(
        "📂 Upload Dataset",
        type=["csv", "xlsx"],
        help="Upload the credit risk CSV or Excel file",
    )

    st.markdown("---")
    st.markdown("**Navigation**")

    # Only two pages now
    page = st.radio(
        "",
        ["📊 Overview", "🎯 Live Prediction"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Credit Risk ML · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# LANDING SCREEN (no file uploaded yet)
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem;">💳</div>
        <h1 style="color: #ccd6f6; font-size: 2.2rem; margin-top: 1rem;">
            Credit Risk Intelligence
        </h1>
        <p style="color: #8892b0; font-size: 1.1rem; max-width: 540px; margin: 1rem auto;">
            Upload your credit risk dataset using the sidebar to explore insights
            and make real-time loan approval predictions.
        </p>
        <div style="margin-top: 2rem; color: #64ffda; font-size: 0.9rem;">
            ← Upload a CSV or XLSX file to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()   # nothing more to show until a file is provided


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset..."):
    df_raw = load_data(uploaded_file)
    df_proc, y_series = preprocess_df(df_raw)

models     = load_models()
has_target = y_series is not None   # True when LoanApproved column exists


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 Dataset Overview")

    # ── KPI row ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(metric_card("Total Applications", f"{len(df_raw):,}"),
                    unsafe_allow_html=True)
    with col2:
        approval_rate = df_raw[TARGET_COL].mean() * 100 if has_target else 0
        st.markdown(metric_card("Approval Rate", f"{approval_rate:.1f}%"),
                    unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Features", str(df_raw.shape[1])),
                    unsafe_allow_html=True)
    with col4:
        missing_pct = (
            df_raw.isnull().sum().sum()
            / (df_raw.shape[0] * df_raw.shape[1]) * 100
        )
        st.markdown(metric_card("Missing Data", f"{missing_pct:.2f}%"),
                    unsafe_allow_html=True)
    with col5:
        avg_income = df_raw["AnnualIncome"].mean() if "AnnualIncome" in df_raw.columns else 0
        st.markdown(metric_card("Avg Annual Income", f"₹{avg_income:,.0f}"),
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Pie chart + Key stats ─────────────────────────────────────────────────
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.markdown('<div class="section-header">Loan Approval Split</div>',
                    unsafe_allow_html=True)
        if has_target:
            counts = df_raw[TARGET_COL].value_counts()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(
                counts.values,
                labels=["Rejected", "Approved"],
                autopct="%1.1f%%",
                colors=["#ff6b6b", "#64ffda"],
                startangle=90,
                explode=(0.04, 0.04),
                textprops={"color": "#ccd6f6", "fontsize": 11},
            )
            fig.patch.set_facecolor("#161b27")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No target column found in the dataset.")

    with col_b:
        st.markdown('<div class="section-header">Key Statistics</div>',
                    unsafe_allow_html=True)
        # Show descriptive stats for important numeric columns
        stat_cols = ["Age", "AnnualIncome", "CreditScore",
                     "LoanAmount", "DebtToIncomeRatio", "NetWorth"]
        stat_cols = [c for c in stat_cols if c in df_raw.columns]
        stats = df_raw[stat_cols].describe().round(2)
        st.dataframe(stats.style.background_gradient(cmap="Blues", axis=1),
                     use_container_width=True)

    # ── Sample rows ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Data (first 10 rows)</div>',
                unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)

    # ── Approval rate by category ─────────────────────────────────────────────
    if has_target:
        st.markdown('<div class="section-header">Approval Rate by Category</div>',
                    unsafe_allow_html=True)

        cat_cols = ["EmploymentStatus", "EducationLevel",
                    "HomeOwnershipStatus", "LoanPurpose"]
        cat_cols = [c for c in cat_cols if c in df_raw.columns]

        if cat_cols:
            fig, axes = plt.subplots(1, len(cat_cols),
                                     figsize=(5 * len(cat_cols), 4))
            # Ensure axes is always a list, even when there's only 1 column
            if len(cat_cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, cat_cols):
                rate = (
                    df_raw.groupby(col)[TARGET_COL]
                    .mean()
                    .sort_values(ascending=False) * 100
                )
                bars = ax.bar(rate.index, rate.values,
                              color="#64ffda", edgecolor="#0a3b2e", alpha=0.85)
                ax.set_title(col, fontsize=10, fontweight="bold")
                ax.set_ylabel("Approval Rate (%)")
                ax.tick_params(axis="x", rotation=30)

                # Label each bar with its percentage
                for bar in bars:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}%",
                        ha="center", va="bottom", fontsize=8, color="#ccd6f6",
                    )
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – LIVE PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.markdown("# 🎯 Live Loan Approval Prediction")

    # Guard: models must be trained and saved first
    if not models:
        st.warning(
            "⚠️ No trained models found in the `/models/` folder.  \n"
            "Please run the notebook first to train and save models, then reload this page."
        )
        st.stop()

    # ── Two-column layout: form on left, result on right ─────────────────────
    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-header">Applicant Information</div>',
                    unsafe_allow_html=True)

        # All inputs sit inside a Streamlit form so nothing is sent until the
        # user clicks the button.
        with st.form("prediction_form"):

            # ── Personal ─────────────────────────────────────────────────────
            st.markdown("**👤 Personal Details**")
            c1, c2 = st.columns(2)
            age            = c1.number_input("Age", 18, 80, 35)
            marital_status = c2.selectbox("Marital Status",
                                          ["Single", "Married", "Divorced", "Widowed"])
            num_dependents = c1.number_input("Dependents", 0, 10, 1)
            age_group      = c2.selectbox("Age Group",
                                          ["Young Adult", "Adult", "Middle-Aged", "Senior"])

            # ── Financial ────────────────────────────────────────────────────
            st.markdown("**💰 Financial Details**")
            c3, c4 = st.columns(2)
            annual_income  = c3.number_input("Annual Income (₹)",  5_000,  500_000, 60_000)
            monthly_income = c4.number_input("Monthly Income (₹)",   500,   50_000,  5_000)
            loan_amount    = c3.number_input("Loan Amount (₹)",    1_000,  500_000, 20_000)
            loan_duration  = c4.number_input("Loan Duration (months)", 6, 120, 48)

            # ── Credit profile ───────────────────────────────────────────────
            st.markdown("**📊 Credit Profile**")
            c5, c6 = st.columns(2)
            credit_score    = c5.slider("Credit Score", 300, 850, 650)
            dti             = c6.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35)
            credit_util     = c5.slider("Credit Card Utilisation", 0.0, 1.0, 0.30)
            num_credit_lines= c6.number_input("Open Credit Lines", 0, 20, 3)
            num_inquiries   = c5.number_input("Credit Inquiries", 0, 20, 2)
            length_credit   = c6.number_input("Credit History Length (yrs)", 0, 30, 8)
            payment_history = c5.number_input("Payment History Score", 0, 50, 25)
            prev_defaults   = c6.selectbox("Previous Loan Defaults", [0, 1, 2, 3])
            bankruptcy      = st.selectbox("Bankruptcy History", [0, 1])

            # ── Employment & housing ─────────────────────────────────────────
            st.markdown("**🏠 Employment & Housing**")
            c7, c8 = st.columns(2)
            employment     = c7.selectbox("Employment Status",
                                          ["Employed", "Self-Employed", "Unemployed", "Part-Time"])
            education      = c8.selectbox("Education Level",
                                          ["High School", "Associate", "Bachelor", "Master", "PhD"])
            job_tenure     = c7.number_input("Job Tenure (years)", 0, 40, 5)
            home_ownership = c8.selectbox("Home Ownership", ["Own", "Rent", "Mortgage"])
            loan_purpose   = c7.selectbox("Loan Purpose",
                                          ["Home", "Auto", "Education",
                                           "Debt Consolidation", "Business", "Medical", "Personal"])

            # ── Assets & liabilities ─────────────────────────────────────────
            st.markdown("**🏦 Assets & Liabilities**")
            c9, c10 = st.columns(2)
            savings          = c9.number_input("Savings Balance (₹)",   0, 500_000,  8_000)
            checking         = c10.number_input("Checking Balance (₹)", 0, 200_000,  2_000)
            total_assets     = c9.number_input("Total Assets (₹)",      0, 5_000_000, 100_000)
            total_liabilities= c10.number_input("Total Liabilities (₹)",0, 2_000_000,  20_000)
            net_worth        = total_assets - total_liabilities   # auto-calculated
            monthly_debt     = c9.number_input("Monthly Debt Payments (₹)", 0, 10_000, 500)

            # ── Segment fields ───────────────────────────────────────────────
            st.markdown("**🏷️ Derived / Segment Fields**")
            c11, c12 = st.columns(2)
            credit_tier      = c11.selectbox("Credit Tier",
                                             ["1. Poor", "2. Fair", "3. Good",
                                              "4. Very Good", "5. Excellent"])
            risk_band        = c12.selectbox("Risk Band",
                                             ["Low Risk", "Medium Risk", "High Risk"])
            income_bracket   = c11.selectbox("Income Bracket",
                                             ["Low", "Medium", "High", "Very High"])
            dti_bucket       = c12.selectbox("DTI Bucket",
                                             ["Healthy (<30%)", "Moderate (30-40%)",
                                              "High (40-50%)", "Very High (>50%)"])
            income_stability = c11.selectbox("Income Stability", ["Stable", "Unstable"])
            risk_segment     = c12.selectbox("Risk Segment",
                                             ["Normal", "Low Risk (Prime)", "High Risk (Subprime)"])

            # ── Model choice ─────────────────────────────────────────────────
            model_choice = st.selectbox("🤖 Prediction Model", list(models.keys()))

            submitted = st.form_submit_button("🚀 Predict Approval",
                                              use_container_width=True)

    # ── Result panel ──────────────────────────────────────────────────────────
    with col_result:
        if submitted:
            # ── Build the feature dict that matches training features ─────────
            # Each key is a column name the model was trained on.
            input_data = {
                "Age":                      age,
                "AnnualIncome":             annual_income,
                "CreditScore":              credit_score,
                "EmploymentStatus":         employment,      # encoded below
                "EducationLevel":           education,       # encoded via ORDINAL_MAPS
                "Experience":               job_tenure,
                "LoanAmount":               loan_amount,
                "LoanDuration":             loan_duration,
                "MaritalStatus":            marital_status,  # encoded below
                "NumberOfDependents":       num_dependents,
                "HomeOwnershipStatus":      home_ownership,  # encoded below
                "MonthlyDebtPayments":      monthly_debt,
                "CreditCardUtilizationRate":credit_util,
                "NumberOfOpenCreditLines":  num_credit_lines,
                "NumberOfCreditInquiries":  num_inquiries,
                "DebtToIncomeRatio":        dti,
                "BankruptcyHistory":        bankruptcy,
                "LoanPurpose":              loan_purpose,    # encoded below
                "PreviousLoanDefaults":     prev_defaults,
                "PaymentHistory":           payment_history,
                "LengthOfCreditHistory":    length_credit,
                "SavingsAccountBalance":    savings,
                "CheckingAccountBalance":   checking,
                "TotalAssets":              total_assets,
                "TotalLiabilities":         total_liabilities,
                "MonthlyIncome":            monthly_income,
                "UtilityBillsPaymentHistory": 0.8,          # sensible default
                "JobTenure":                job_tenure,
                "NetWorth":                 net_worth,
                "BaseInterestRate":         0.20,
                "TotalDebtToIncomeRatio":   dti,
                "RiskScore":                50,
                "App_Quarter":              0,
                "MonthlyObligations":       monthly_debt + loan_amount / loan_duration,
                "ResidualIncome":           monthly_income - monthly_debt,
                # Ordinal-encoded fields
                "Income_Stability":   1 if income_stability == "Stable" else 0,
                "Credit_Tier":        ORDINAL_MAPS["Credit_Tier"].get(credit_tier, 1),
                "Risk_Band":          ORDINAL_MAPS["Risk_Band"].get(risk_band, 1),
                "DTI_Bucket":         ORDINAL_MAPS["DTI_Bucket"].get(dti_bucket, 1),
                "Risk_Segment":       (0 if "Normal" in risk_segment
                                       else 1 if "Prime" in risk_segment else 2),
                "Income_Bracket":     ORDINAL_MAPS["Income_Bracket"].get(income_bracket, 1),
                "AgeGroup":           ORDINAL_MAPS["AgeGroup"].get(age_group, 1),
                "Purpose_Default_Rate": 0.09,
            }

            # Apply label encoding to remaining categorical columns
            for col in LABEL_MAPS:
                if col in input_data:
                    input_data[col] = LABEL_MAPS[col].get(input_data[col], 0)

            # Education level is in ORDINAL_MAPS but stored as string above
            input_data["EducationLevel"] = ORDINAL_MAPS["EducationLevel"].get(education, 2)

            # Convert dict → single-row DataFrame
            input_df = pd.DataFrame([input_data])

            model_obj = models[model_choice]

            # Align columns to whatever the model was trained on
            if hasattr(model_obj, "feature_names_in_"):
                feat_names = list(model_obj.feature_names_in_)
                # Add any missing columns with value 0
                for f in feat_names:
                    if f not in input_df.columns:
                        input_df[f] = 0
                input_df = input_df[feat_names]   # reorder to match training

            # ── Run prediction ────────────────────────────────────────────────
            try:
                prob = model_obj.predict_proba(input_df)[0][1]  # P(Approved)
                pred = int(prob >= 0.5)                          # 1 = Approved

                # Result badge
                st.markdown('<div class="section-header">Prediction Result</div>',
                            unsafe_allow_html=True)

                if pred == 1:
                    st.markdown(
                        f'<div class="pred-approved">✅ LOAN APPROVED'
                        f'<br><small>Approval Probability: {prob*100:.1f}%</small></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="pred-rejected">❌ LOAN REJECTED'
                        f'<br><small>Approval Probability: {prob*100:.1f}%</small></div>',
                        unsafe_allow_html=True,
                    )

                # Risk label
                risk_pct = prob * 100
                if risk_pct >= 70:
                    risk_class, risk_label = "risk-low",  "✅ Low Risk Applicant"
                elif risk_pct >= 40:
                    risk_class, risk_label = "risk-mid",  "⚠️ Moderate Risk"
                else:
                    risk_class, risk_label = "risk-high", "🔴 High Risk Applicant"

                st.markdown(
                    f'<p class="{risk_class}" style="font-size:1.1rem; margin-top:1rem;">'
                    f'{risk_label}</p>',
                    unsafe_allow_html=True,
                )

                # Probability bar chart
                fig, ax = plt.subplots(figsize=(7, 1.3))
                bar_color = "#64ffda" if pred == 1 else "#ff6b6b"
                ax.barh([0], [prob],       color=bar_color, height=0.5, edgecolor="#21262d")
                ax.barh([0], [1 - prob],   left=[prob], color="#21262d", height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
                ax.set_yticks([])
                ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
                ax.set_title(f"Approval Probability: {prob*100:.1f}%",
                             fontweight="bold", fontsize=11)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Key risk factors summary ──────────────────────────────────
                st.markdown('<div class="section-header">Key Risk Factors</div>',
                            unsafe_allow_html=True)

                # Each tuple is  (icon, message)
                factors = []

                if credit_score < 580:
                    factors.append(("🔴", f"Credit score {credit_score} is below 580 — Poor tier"))
                elif credit_score > 720:
                    factors.append(("🟢", f"Strong credit score: {credit_score}"))

                if dti > 0.43:
                    factors.append(("🔴", f"DTI {dti:.2f} exceeds safe threshold (0.43)"))
                else:
                    factors.append(("🟢", f"DTI {dti:.2f} is within acceptable range"))

                if prev_defaults > 0:
                    factors.append(("🔴", f"{prev_defaults} previous loan default(s) on record"))

                if bankruptcy == 1:
                    factors.append(("🔴", "Bankruptcy history detected"))

                if annual_income > 60_000:
                    factors.append(("🟢", f"Strong annual income: ₹{annual_income:,}"))

                if net_worth > 50_000:
                    factors.append(("🟢", f"Positive net worth: ₹{net_worth:,}"))

                for icon, text in factors:
                    st.markdown(f"**{icon}** {text}")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure the model was trained with the same features.  "
                        "Check feature alignment in the notebook.")

        else:
            # Placeholder shown before the user submits
            st.markdown("""
            <div style="text-align:center; padding: 4rem 1rem; color: #8892b0;">
                <div style="font-size: 3rem;">🎯</div>
                <p style="font-size: 1rem; margin-top: 1rem;">
                    Fill in the applicant details on the left<br>
                    and click <strong>Predict Approval</strong> to get results.
                </p>
            </div>
            """, unsafe_allow_html=True)
