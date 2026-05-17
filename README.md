# ChurnIQ — Customer Intelligence Platform

> An AI-powered customer churn prediction platform built with Machine Learning, Explainable AI, and Claude AI — designed for real-world telecom business use cases.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green) ![Claude AI](https://img.shields.io/badge/Claude-AI_Summaries-purple)

---

## Overview

ChurnIQ predicts whether a telecom customer is likely to cancel their subscription using a Random Forest classifier trained on 7,043 real customer records. It goes beyond a basic ML model by adding SHAP explainability, AI-generated analyst summaries powered by Claude, customer persona detection, and revenue impact estimation — making it a complete business intelligence tool.

---

## Features

- **Real-time Churn Prediction** — Enter customer details and get an instant churn probability with a dark-themed risk gauge
- **SHAP Explainability** — Visual feature contribution chart showing exactly why a customer is at risk
- **AI Analyst Summary** — Claude AI generates a 3-sentence plain-English business insight per customer
- **Customer Persona Detection** — Automatically classifies customers into 6 archetypes (New At-Risk, Long-term Loyal, Price-Sensitive, etc.)
- **Revenue Impact Estimator** — Calculates annual revenue at risk and ROI of retention offers
- **Retention Recommendations** — Actionable, data-driven suggestions tailored to each customer's risk factors
- **Batch Prediction** — Upload a CSV of hundreds of customers, get predictions for all, download results
- **Session History** — Tracks all predictions made in the current session

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Random Forest Classifier (scikit-learn) |
| Explainability | SHAP (SHapley Additive exPlanations) |
| AI Summaries | Anthropic Claude (claude-sonnet-4) |
| Frontend | Streamlit |
| Visualization | Plotly, Matplotlib |
| Language | Python 3.11 |
| Dataset | IBM Telco Customer Churn (7,043 records) |

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 79.28% |
| Precision (Churn class) | 65% |
| Recall (Churn class) | 48% |
| F1 Score (Churn class) | 0.55 |

---

## Project Structure

```
churn-predictor/
├── app.py          # Streamlit web application
├── model.py        # ML training pipeline
├── .gitignore
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/shashwatsharma028-sys/churniq-customer-intelligence.git
cd churniq-customer-intelligence
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn shap streamlit plotly matplotlib anthropic
```

**3. Download the dataset**

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place it at:
```
data/telco_churn.csv
```

**4. Train the model**
```bash
python model.py
```
This generates `churn_model.pkl` and `feature_names.pkl`.

**5. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**6. Run the app**
```bash
streamlit run app.py
```

---

## How It Works

1. **Data Pipeline** — Loads raw Telco CSV, cleans missing values, encodes categoricals, engineers 4 new features
2. **Model Training** — Trains a Random Forest with 200 estimators on 80% of data, evaluates on 20%
3. **SHAP Explainability** — TreeExplainer computes feature contributions per prediction
4. **AI Summary** — Sends prediction context to Claude API, returns plain-English business insight
5. **Streamlit UI** — Dark-themed professional dashboard with real-time predictions

---

## Business Use Case

This tool addresses a real enterprise problem — telecom companies lose billions annually to customer churn. ChurnIQ helps customer success teams:
- Identify at-risk customers before they cancel
- Understand the specific reasons driving each customer's risk
- Prioritize retention efforts based on revenue impact
- Take targeted action with AI-recommended retention strategies

This is exactly the type of solution TCS builds for its telecom clients worldwide.

---

## Author

**Shashwat Sharma**
AI/ML Enthusiast | B.Tech Student

---

*Built as part of an AI/ML portfolio for industry applications*
