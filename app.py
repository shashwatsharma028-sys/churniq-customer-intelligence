import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import anthropic
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ChurnIQ — Customer Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  #MainMenu, footer { visibility: hidden; }

  section[data-testid="stSidebar"] { background: #0f172a; }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  div[data-testid="metric-container"] {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 10px; padding: 16px;
  }

  .stButton > button {
    background: #1e40af; color: white; border: none;
    border-radius: 8px; padding: 10px 24px;
    font-weight: 500; font-size: 14px; width: 100%;
  }
  .stButton > button:hover { background: #1d4ed8; }

  .section-header {
    font-size: 12px; font-weight: 600; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin: 24px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
  }

  .pro-card {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 12px; padding: 22px 26px; margin-bottom: 16px;
  }

  .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .badge-high   { background: #450a0a; color: #fca5a5; }
  .badge-medium { background: #451a03; color: #fcd34d; }
  .badge-low    { background: #052e16; color: #86efac; }

  .dark-panel {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 14px; padding: 24px; margin-bottom: 16px;
  }

  .dark-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .dark-table td {
    padding: 16px 12px;
    vertical-align: middle;
    border-bottom: 1px solid #1e293b;
    line-height: 1.6;
  }
  .dark-table td:first-child { color: #94a3b8; text-align: left; }
  .dark-table td:last-child  { color: #e2e8f0; font-weight: 600; text-align: right; }
  .dark-table tr:last-child td {
    border-bottom: none; color: #38bdf8;
    font-size: 14px; font-weight: 700; padding-top: 20px;
  }

  .dark-ai-box {
    background: #0f172a; border: 1px solid #1e293b;
    border-radius: 14px; padding: 22px 24px; margin-bottom: 16px;
  }
  .dark-ai-label {
    font-size: 11px; font-weight: 600; color: #38bdf8;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;
  }
  .dark-ai-text { font-size: 14px; color: #cbd5e1; line-height: 1.8; }

  .dark-suggestion {
    background: #0f172a; border: 1px solid #1e293b;
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; font-size: 13px;
    color: #94a3b8; margin-bottom: 8px; line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_names.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, feature_names = load_model()
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

CAT_MAP = {
    "gender":           {"Male": 1, "Female": 0},
    "Partner":          {"Yes": 1, "No": 0},
    "Dependents":       {"Yes": 1, "No": 0},
    "PhoneService":     {"Yes": 1, "No": 0},
    "MultipleLines":    {"Yes": 2, "No": 1, "No phone service": 0},
    "InternetService":  {"Fiber optic": 1, "DSL": 0, "No": 2},
    "OnlineSecurity":   {"Yes": 2, "No": 1, "No internet service": 0},
    "OnlineBackup":     {"Yes": 2, "No": 1, "No internet service": 0},
    "DeviceProtection": {"Yes": 2, "No": 1, "No internet service": 0},
    "TechSupport":      {"Yes": 2, "No": 1, "No internet service": 0},
    "StreamingTV":      {"Yes": 2, "No": 1, "No internet service": 0},
    "StreamingMovies":  {"Yes": 2, "No": 1, "No internet service": 0},
    "Contract":         {"Month-to-month": 0, "One year": 1, "Two year": 2},
    "PaperlessBilling": {"Yes": 1, "No": 0},
    "PaymentMethod":    {
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    },
}

def build_input(d):
    row = {}
    for feat in feature_names:
        if feat in CAT_MAP:
            row[feat] = CAT_MAP[feat].get(d.get(feat, list(CAT_MAP[feat].keys())[0]), 0)
        elif feat == "AvgMonthlyCharge":
            row[feat] = d["TotalCharges"] / (d["tenure"] + 1)
        elif feat == "IsNewCustomer":
            row[feat] = int(d["tenure"] < 6)
        elif feat == "IsLongTermCustomer":
            row[feat] = int(d["tenure"] > 36)
        elif feat == "ChargePerService":
            row[feat] = d["MonthlyCharges"] / (d["tenure"] + 1)
        else:
            row[feat] = d.get(feat, 0)
    return pd.DataFrame([row])[feature_names]

def detect_persona(d, prob):
    tenure   = d.get("tenure", 0)
    monthly  = d.get("MonthlyCharges", 0)
    contract = d.get("Contract", "")
    if tenure > 36 and prob < 0.4:
        return "Long-term Loyal Customer", "#166534", "#dcfce7"
    if tenure < 6 and prob > 0.5:
        return "New At-Risk Customer", "#991b1b", "#fee2e2"
    if monthly > 80 and prob > 0.5:
        return "Price-Sensitive User", "#92400e", "#fef3c7"
    if contract == "Month-to-month" and prob > 0.5:
        return "Uncommitted Subscriber", "#5b21b6", "#ede9fe"
    if prob < 0.3:
        return "Satisfied Stable Customer", "#1e40af", "#eff6ff"
    return "Moderate Risk Customer", "#374151", "#f1f5f9"

def get_suggestions(d, prob):
    if prob < 0.4:
        return ["This customer shows low churn risk. Continue standard engagement practices and monitor quarterly."]
    tips = []
    if d.get("Contract") == "Month-to-month":
        tips.append("Offer a discounted annual or two-year contract to increase commitment and reduce churn probability.")
    if d.get("tenure", 0) < 12:
        tips.append("Assign a dedicated onboarding specialist — early tenure customers are highest risk.")
    if d.get("MonthlyCharges", 0) > 70:
        tips.append("Apply a 3-month billing discount of 10-15% to reduce price sensitivity.")
    if d.get("TechSupport") == "No":
        tips.append("Proactively offer complimentary tech support — customers without support churn at 2x the rate.")
    if d.get("PaymentMethod") == "Electronic check":
        tips.append("Encourage migration to automatic payment — reduces churn by approximately 10% across cohorts.")
    if not tips:
        tips.append("Schedule a proactive customer success call to identify unstated concerns.")
    return tips

def revenue_impact(d):
    monthly = d.get("MonthlyCharges", 0)
    return monthly * 12, monthly * 24, monthly * 3 * 0.15

def gauge_chart(prob):
    color = "#22c55e" if prob < 0.4 else "#f59e0b" if prob < 0.7 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "color": color, "family": "Inter"}},
        title={"text": "CHURN PROBABILITY", "font": {"size": 11, "color": "#475569", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0f172a",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#052e16"},
                {"range": [40, 70], "color": "#2d1a00"},
                {"range": [70,100], "color": "#2d0000"},
            ],
            "threshold": {"line": {"color": color, "width": 2}, "thickness": 0.8, "value": prob * 100}
        }
    ))
    fig.update_layout(
        height=260, margin=dict(t=50, b=10, l=20, r=20),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font={"family": "Inter, sans-serif"}
    )
    return fig

def shap_chart(input_df):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(input_df)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        vals = sv[0, :, 1]
    elif isinstance(sv, list):
        vals = np.array(sv[1])[0]
    else:
        vals = sv[0]

    shap_df = pd.DataFrame({
        "Feature": input_df.columns.tolist(),
        "Impact":  vals
    }).sort_values("Impact", key=abs, ascending=True).tail(8)

    colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["Impact"].tolist()]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.barh(shap_df["Feature"].tolist(), shap_df["Impact"].tolist(), color=colors, height=0.55)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Impact on churn probability", fontsize=10, color="#475569")
    ax.set_title("Feature Contribution Analysis", fontsize=12, color="#e2e8f0", fontweight="600", pad=14)
    ax.tick_params(colors="#64748b", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    return fig

def ai_explanation(name, prob, top_features, persona, suggestions):
    risk_level = "high" if prob > 0.7 else "moderate" if prob > 0.4 else "low"
    prompt = f"""You are a senior customer success analyst writing a concise professional insight for a business manager.

Customer: {name}
Churn probability: {prob*100:.1f}%
Risk level: {risk_level}
Customer persona: {persona}
Top risk factors: {', '.join(top_features[:3])}
Recommended action: {suggestions[0] if suggestions else 'Standard monitoring'}

Write exactly 3 sentences in plain business English. No bullet points, no emojis, no markdown:
1. State the risk level and the primary factor driving it.
2. Quantify the business impact if this customer churns.
3. State the single most important action the team should take immediately.

Be direct, specific, and professional."""
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=220,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except:
        return f"Risk assessment: {prob*100:.1f}% churn probability detected. Primary driver is {top_features[0] if top_features else 'tenure and billing patterns'}. Immediate retention action is recommended based on the factors outlined below."

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### ChurnIQ")
    st.markdown("<p style='font-size:12px;color:#475569;margin-top:-10px;'>Customer Intelligence Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["Single Prediction", "Batch Analysis", "About"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<p style='font-size:11px;color:#475569;'>Model: Random Forest<br>Dataset: IBM Telco — 7,043 records<br>Accuracy: 79.28%</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# ── PAGE 1 ──
if page == "Single Prediction":
    st.markdown("## Customer Churn Analysis")
    st.markdown("<p style='color:#64748b;margin-top:-12px;'>Enter customer details to generate a risk assessment.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("churn_form"):
        customer_name = st.text_input("Customer Name", placeholder="e.g. Rahul Sharma")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='section-header'>Demographics</div>", unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Male", "Female"])
            senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner    = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("<div class='section-header'>Services</div>", unsafe_allow_html=True)
            phone    = st.selectbox("Phone Service", ["Yes", "No"])
            multi    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech     = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv       = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.markdown("<div class='section-header'>Billing</div>", unsafe_allow_html=True)
            contract  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment   = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
            total   = st.number_input("Total Charges ($)", 0.0, 10000.0, 1500.0)

        submitted = st.form_submit_button("Generate Risk Assessment", use_container_width=True)

    if submitted:
        name = customer_name.strip() if customer_name.strip() else "This customer"
        d = {
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total
        }

        input_df = build_input(d)
        prob     = model.predict_proba(input_df)[0][1]
        risk_label = "High Risk" if prob >= 0.7 else "Moderate Risk" if prob >= 0.4 else "Low Risk"
        risk_class = "badge-high" if prob >= 0.7 else "badge-medium" if prob >= 0.4 else "badge-low"
        persona, persona_color, persona_bg = detect_persona(d, prob)
        suggestions = get_suggestions(d, prob)
        annual, ltv, discount_cost = revenue_impact(d)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_df)
        if isinstance(sv, np.ndarray) and sv.ndim == 3:
            vals = sv[0, :, 1]
        elif isinstance(sv, list):
            vals = np.array(sv[1])[0]
        else:
            vals = sv[0]
        top_features = [feature_names[i] for i in np.argsort(np.abs(vals))[::-1][:3]]

        st.session_state.history.append({
            "Customer": name, "Tenure (mo)": tenure, "Contract": contract,
            "Monthly ($)": monthly, "Risk Score": f"{prob*100:.1f}%", "Assessment": risk_label
        })

        st.markdown("---")

        # Customer Profile
        st.markdown("<div class='section-header'>Customer Profile</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='pro-card'>
          <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
            <div>
              <h2 style='margin:0 0 6px;font-size:22px;color:#f1f5f9;font-weight:700;'>{name}</h2>
              <div style='background:{persona_bg};color:{persona_color};display:inline-block;
                          padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;'>
                {persona}
              </div>
            </div>
            <div class='badge {risk_class}'>{risk_label}</div>
          </div>
          <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
                      margin-top:18px;padding-top:16px;border-top:1px solid #1e293b;'>
            <div>
              <div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.05em;'>Tenure</div>
              <div style='font-size:18px;font-weight:600;color:#e2e8f0;'>{tenure} months</div>
            </div>
            <div>
              <div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.05em;'>Contract</div>
              <div style='font-size:18px;font-weight:600;color:#e2e8f0;'>{contract}</div>
            </div>
            <div>
              <div style='font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.05em;'>Monthly Charges</div>
              <div style='font-size:18px;font-weight:600;color:#e2e8f0;'>${monthly:.0f}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Results Row — NO duplicate headers
        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div class='dark-panel'>
              <div style='font-size:12px;font-weight:600;color:#475569;text-transform:uppercase;
                          letter-spacing:0.07em;margin-bottom:16px;padding-bottom:10px;
                          border-bottom:1px solid #1e293b;'>Revenue at Risk</div>
              <table class='dark-table'>
                <tr><td>Annual recurring revenue</td><td>${annual:,.0f}</td></tr>
                <tr><td>Estimated lifetime value (24 mo)</td><td>${ltv:,.0f}</td></tr>
                <tr><td>Cost of 3-month retention offer</td><td>${discount_cost:,.0f}</td></tr>
                <tr><td><strong>Net savings if retained</strong></td><td>${annual - discount_cost:,.0f}</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)

        with c2:
            fig = shap_chart(input_df)
            st.pyplot(fig)
            plt.close()

        # AI Summary
        st.markdown("<div class='section-header'>Analyst Summary</div>", unsafe_allow_html=True)
        with st.spinner("Generating analyst summary..."):
            explanation = ai_explanation(name, prob, top_features, persona, suggestions)
        st.markdown(f"""
        <div class='dark-ai-box'>
          <div class='dark-ai-label'>AI Analyst — Powered by Claude</div>
          <div class='dark-ai-text'>{explanation}</div>
        </div>""", unsafe_allow_html=True)

        # Retention Recommendations
        st.markdown("<div class='section-header'>Retention Recommendations</div>", unsafe_allow_html=True)
        for tip in suggestions:
            st.markdown(f"<div class='dark-suggestion'>{tip}</div>", unsafe_allow_html=True)

        # History
        if st.session_state.history:
            st.markdown("<div class='section-header'>Session History</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

# ── PAGE 2 ──
elif page == "Batch Analysis":
    st.markdown("## Batch Customer Analysis")
    st.markdown("<p style='color:#64748b;margin-top:-12px;'>Upload a customer dataset to generate churn predictions at scale.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Upload a CSV in the standard Telco format (same columns as the training dataset).")
    uploaded = st.file_uploader("Select CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df):,} records loaded**")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)

        if st.button("Run Batch Analysis", use_container_width=True):
            results  = []
            progress = st.progress(0)
            for i, (_, row) in enumerate(df.iterrows()):
                try:
                    inp = build_input(row.to_dict())
                    p   = model.predict_proba(inp)[0][1]
                    results.append({
                        "Churn Probability": f"{p*100:.1f}%",
                        "Risk Assessment":   "High Risk" if p >= 0.7 else "Moderate Risk" if p >= 0.4 else "Low Risk",
                        "Predicted Outcome": "Churn" if p >= 0.5 else "Retain"
                    })
                except:
                    results.append({"Churn Probability": "Error", "Risk Assessment": "Error", "Predicted Outcome": "Error"})
                progress.progress((i + 1) / len(df))

            result_df   = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
            churn_count = sum(1 for r in results if r["Predicted Outcome"] == "Churn")
            high_risk   = sum(1 for r in results if r["Risk Assessment"] == "High Risk")

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers",    f"{len(results):,}")
            m2.metric("Predicted to Churn", f"{churn_count:,}")
            m3.metric("High Risk",          f"{high_risk:,}")
            m4.metric("Churn Rate",         f"{churn_count/len(results)*100:.1f}%")

            st.dataframe(result_df, use_container_width=True, hide_index=True)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results as CSV", csv,
                               "churn_analysis_results.csv", "text/csv",
                               use_container_width=True)

# ── PAGE 3 ──
elif page == "About":
    st.markdown("## About ChurnIQ")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Project Overview
        ChurnIQ is a customer intelligence platform that predicts churn risk for telecom customers
        using machine learning and explainable AI. It combines a Random Forest classifier with
        SHAP-based feature attribution and Claude AI to deliver actionable, human-readable insights.

        ### Technical Stack
        - **Model:** Random Forest Classifier (scikit-learn)
        - **Explainability:** SHAP (SHapley Additive exPlanations)
        - **AI Summaries:** Anthropic Claude
        - **Frontend:** Streamlit
        - **Visualization:** Plotly, Matplotlib
        - **Dataset:** IBM Telco Customer Churn — 7,043 records, 21 features
        """)
    with col2:
        st.markdown("""
        ### Model Performance
        | Metric | Score |
        |---|---|
        | Accuracy | 79.28% |
        | Precision (Churn) | 65% |
        | Recall (Churn) | 48% |
        | F1 Score (Churn) | 0.55 |

        ### Key Features
        - Real-time churn probability with dark-themed risk gauge
        - SHAP feature contribution analysis per customer
        - Customer persona detection (6 archetypes)
        - Revenue impact and retention ROI calculation
        - AI analyst summary powered by Claude
        - Batch prediction with CSV upload and download
        """)
    st.markdown("---")
    st.markdown("<p style='color:#64748b;font-size:13px;'>Built by Shashwat Sharma &nbsp;|&nbsp; AI/ML Portfolio Project &nbsp;|&nbsp; 2025</p>", unsafe_allow_html=True)