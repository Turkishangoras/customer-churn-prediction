import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
# Sets the browser tab title, icon, and page width behavior.
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="centered"
)

# -------------------------
# CUSTOM CSS (Premium UI)
# -------------------------
# This block customizes the Streamlit app with a premium dark theme.
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #050816 0%, #0b1020 45%, #111827 100%);
        color: #f9fafb;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 1.8rem;
    }

    .section-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.2rem 1.2rem 0.6rem 1.2rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.8rem;
    }

    .result-box {
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.05rem;
    }

    .risk-high {
        background: linear-gradient(90deg, rgba(127,29,29,0.95), rgba(220,38,38,0.80));
        border: 1px solid rgba(248,113,113,0.35);
        color: #fff7f7;
    }

    .risk-low {
        background: linear-gradient(90deg, rgba(20,83,45,0.95), rgba(22,163,74,0.75));
        border: 1px solid rgba(74,222,128,0.35);
        color: #f0fdf4;
    }

    .small-note {
        color: #cbd5e1;
        font-size: 0.95rem;
    }

    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 700;
        font-size: 1rem;
        color: white;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
    }

    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9);
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 2rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD MODEL
# -------------------------
# Loads the trained machine learning pipeline from disk.
# This pipeline already includes preprocessing + Random Forest model.
model = joblib.load("outputs/models/churn_model.joblib")


@st.cache_data
def load_base_data():
    """
    Load and cache the original dataset structure.

    Why this is useful:
    - The model expects the same feature columns used during training.
    - We use one base row from the dataset, then replace fields with user input.
    - Caching prevents reloading the CSV every time the user changes something.
    """
    df = pd.read_csv("data/Telco_customer_churn.csv")
    df.columns = df.columns.str.strip()  # remove accidental spaces in column names
    return df


@st.cache_data
def get_feature_importance_df():
    """
    Extract feature names and feature importance values from the trained pipeline.

    Returns:
        A dataframe with the top 10 most important features, sorted for a horizontal bar chart.
    """
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    feat_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Get top 10 highest importance features
    feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False).head(10)

    # Sort ascending so the horizontal chart looks cleaner
    feat_imp_df = feat_imp_df.sort_values("Importance", ascending=True)

    return feat_imp_df


# Load dataset structure once
df = load_base_data()

# -------------------------
# HEADER
# -------------------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict customer churn risk using a trained machine learning model with interactive business inputs.</div>',
    unsafe_allow_html=True
)

# -------------------------
# CUSTOMER PROFILE
# -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Basic customer attributes
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    partner = st.selectbox("Partner", ["Yes", "No"])

with col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# SERVICES
# -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

with col4:
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# BILLING
# -------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Billing</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col6:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# BUILD INPUT DATA
# -------------------------
# Start with one correctly shaped row from the original training data.
# Then replace the values with what the user selected in the UI.
input_data = df.drop(columns=["Churn Value"]).iloc[0:1].copy()

updates = {
    "Gender": gender,
    "Senior Citizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure Months": tenure,
    "Phone Service": phone_service,
    "Multiple Lines": multiple_lines,
    "Internet Service": internet_service,
    "Online Security": online_security,
    "Online Backup": online_backup,
    "Device Protection": device_protection,
    "Tech Support": tech_support,
    "Streaming TV": streaming_tv,
    "Streaming Movies": streaming_movies,
    "Contract": contract,
    "Paperless Billing": paperless_billing,
    "Payment Method": payment_method,
    "Monthly Charges": monthly_charges,
    "Total Charges": total_charges,
}

# Only update columns that exist in the dataset
for col, value in updates.items():
    if col in input_data.columns:
        input_data[col] = value

# -------------------------
# PREDICT BUTTON
# -------------------------
if st.button("Predict Churn Risk"):
    # Predict the class:
    # 0 = likely to stay
    # 1 = likely to churn
    prediction = model.predict(input_data)[0]

    # Predict the churn probability (class 1)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### Prediction Result")

    # Show styled result banner
    if prediction == 1:
        st.markdown(
            f'<div class="result-box risk-high">⚠️ Customer likely to churn &nbsp;|&nbsp; Probability: {probability:.2%}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box risk-low">✅ Customer likely to stay &nbsp;|&nbsp; Probability: {probability:.2%}</div>',
            unsafe_allow_html=True
        )

    # Risk summary card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Summary</div>', unsafe_allow_html=True)

    col7, col8 = st.columns(2)
    with col7:
        st.metric("Churn Probability", f"{probability:.2%}")
    with col8:
        st.metric("Stay Probability", f"{1 - probability:.2%}")

    st.markdown(
        '<div class="small-note">Key business-style indicators based on the selected profile:</div>',
        unsafe_allow_html=True
    )

    # Business-friendly explanatory hints
    if contract == "Month-to-month":
        st.write("- Month-to-month contracts are commonly associated with higher churn risk.")
    if monthly_charges > 80:
        st.write("- Higher monthly charges may increase churn risk.")
    if tenure < 12:
        st.write("- Shorter customer tenure may increase churn likelihood.")
    if tech_support == "No":
        st.write("- Lack of tech support may contribute to churn.")
    if online_security == "No":
        st.write("- No online security may be associated with higher churn.")
    if payment_method == "Electronic check":
        st.write("- Electronic check payment is often seen in higher-risk churn profiles.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FEATURE IMPORTANCE CHART
# -------------------------
# This section shows an interactive Plotly chart using the model's learned feature importance.
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)

feat_imp_df = get_feature_importance_df()

fig = px.bar(
    feat_imp_df,
    x="Importance",
    y="Feature",
    orientation="h",
    text="Importance",
    title="Top 10 Features Driving Customer Churn"
)

# Style the bars and hover text
fig.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>"
)

# Apply dark dashboard-like styling
fig.update_layout(
    template="plotly_dark",
    height=520,
    title_font_size=20,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=60, b=20),
    xaxis_title="Importance Score",
    yaxis_title="Feature"
)

# Render interactive chart in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    '<div class="small-note">This interactive chart shows which variables had the strongest influence on churn prediction in the trained Random Forest model.</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown(
    '<div class="footer">Built with Streamlit, Plotly, scikit-learn, and a Random Forest churn model.</div>',
    unsafe_allow_html=True
)