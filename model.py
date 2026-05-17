import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import shap
import pickle
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────
def load_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────
# STEP 2: Clean & Preprocess Data
# ─────────────────────────────────────────
def preprocess(df):
    # Drop customerID — not useful for prediction
    df = df.drop(columns=["customerID"])

    # TotalCharges has some spaces — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with median
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target column: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode all categorical columns using LabelEncoder
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────
# STEP 3: Feature Engineering
# ─────────────────────────────────────────
def feature_engineering(df):
    # Create new features that help the model
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
    df["IsLongTermCustomer"] = (df["tenure"] > 36).astype(int)
    df["ChargePerService"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    print(f"\nNew features added. Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────
# STEP 4: Train / Test Split
# ─────────────────────────────────────────
def split_data(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# STEP 5: Train XGBoost Model
# ─────────────────────────────────────────
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("\nModel trained successfully!")
    return model


# ─────────────────────────────────────────
# STEP 6: Evaluate Model
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    return acc


# ─────────────────────────────────────────
# STEP 7: SHAP Explainability
# ─────────────────────────────────────────
def explain_model(model, X_test):
    print("\nCalculating SHAP values (may take 30 seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    print("\nTop 5 features driving churn predictions:")
    # New SHAP returns shape (samples, features, classes) — index [:, :, 1] = churn class
    churn_shap = shap_values[:, :, 1]
    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance": abs(churn_shap).mean(axis=0)
    }).sort_values("importance", ascending=False)

    print(shap_df.head(5).to_string(index=False))
    return explainer, shap_values

# ─────────────────────────────────────────
# STEP 8: Save Model & Feature Names
# ─────────────────────────────────────────
def save_model(model, feature_names):
    with open("churn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("\nModel saved as churn_model.pkl")
    print("Feature names saved as feature_names.pkl")


# ─────────────────────────────────────────
# MAIN — Run all steps
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Customer Churn Predictor — Training Pipeline")
    print("=" * 50)

    df = load_data()
    df = preprocess(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)
    explainer, shap_values = explain_model(model, X_test)

    save_model(model, list(X_train.columns))

    print("\n" + "=" * 50)
    print(f"  Pipeline complete! Final accuracy: {acc * 100:.2f}%")
    print("=" * 50)