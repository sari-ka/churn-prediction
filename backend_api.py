from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import shap
from fastapi.middleware.cors import CORSMiddleware
from retention_engine import get_top_shap_reasons, map_rules_to_actions

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline
pipeline = joblib.load("data/random_forest_model.pkl")

model = pipeline.named_steps["model"]

feature_names = pd.read_csv("data/X_train.csv").columns.tolist()

explainer = shap.TreeExplainer(model)


# ---------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------

def validate_input(data):

    # convert numeric fields
    numeric_fields = [
        "Tenure",
        "OrderCount",
        "DaySinceLastOrder",
        "CashbackAmount",
        "SatisfactionScore",
        "Complain",
        "CouponUsed",
        "CityTier"
    ]

    for field in numeric_fields:
        if field in data:
            data[field] = float(data[field])

    # validation
    if not (0 <= data["Tenure"] <= 60):
        raise ValueError("Tenure must be between 0 and 60")

    if not (1 <= data["SatisfactionScore"] <= 5):
        raise ValueError("SatisfactionScore must be between 1 and 5")

    if not (0 <= data["DaySinceLastOrder"] <= 30):
        raise ValueError("DaySinceLastOrder must be between 0 and 30")

    return data

# ---------------------------------------------------
# FEATURE ENGINEERING (MATCH TRAINING)
# ---------------------------------------------------

def engineer_features(df):

    df["IsNewCustomer"] = (df["Tenure"] <= 3).astype(int)

    df["IsInactive"] = (df["DaySinceLastOrder"] > 7).astype(int)

    df["EngagementScore"] = (
        df["OrderCount"] - df["DaySinceLastOrder"]
    )

    df["OrdersPerTenure"] = df["OrderCount"] / (df["Tenure"] + 1)

    df["ComplaintRisk"] = (
        (df["Complain"] == 1) &
        (df["SatisfactionScore"] <= 3)
    ).astype(int)

    return df


# ---------------------------------------------------
# PREPARE MODEL INPUT
# ---------------------------------------------------

def prepare_model_input(user_input):

    df = pd.DataFrame([user_input])

    df = engineer_features(df)

    df = pd.get_dummies(df)

    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    return df


# ---------------------------------------------------
# PREDICT
# ---------------------------------------------------

@app.post("/predict")

def predict_customer(data: dict):

    data = validate_input(data)

    df = prepare_model_input(data)

    prob = pipeline.predict_proba(df)[0][1]

    churn_prediction = "Yes" if prob >= 0.33 else "No"

    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    else:
        shap_row = shap_values[0]

    top_reasons = get_top_shap_reasons(shap_row, feature_names)

    actions = map_rules_to_actions(top_reasons)

    improved_prob = max(prob - 0.15, 0)

    return {
        "churn_prediction": churn_prediction,
        "churn_probability": float(prob),
        "top_reasons": top_reasons,
        "recommended_actions": actions,
        "new_probability_after_strategy": float(improved_prob)
    }

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):

    # -----------------------------
    # READ CSV
    # -----------------------------
    df = pd.read_csv(file.file)

    # Keep original for reference
    original_df = df.copy()

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df = engineer_features(df)

    # One hot encoding
    df = pd.get_dummies(df)

    # Align columns with training data
    for col in feature_names:
        if col not in df:
            df[col] = 0

    df = df[feature_names]

    # -----------------------------
    # MODEL PREDICTION
    # -----------------------------
    probs = pipeline.predict_proba(df)[:,1]

    # Lower threshold so churn isn't always 0
    predictions = (probs > 0.35).astype(int)

    total_customers = len(df)
    total_churn = int(predictions.sum())
    churn_rate = total_churn / total_customers

    # -----------------------------
    # RISK SEGMENTATION
    # -----------------------------
    high_risk = int((probs >= 0.33).sum())
    medium_risk = int(((probs >= 0.2) & (probs < 0.33)).sum())
    low_risk = int((probs <= 0.2).sum())

    risk_distribution = {
        "high": high_risk,
        "medium": medium_risk,
        "low": low_risk
    }

    # -----------------------------
    # SHAP ANALYSIS
    # -----------------------------
    shap_values = explainer.shap_values(df)

    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    mean_shap = shap_matrix.mean(axis=0)

    top_reasons = get_top_shap_reasons(mean_shap, feature_names)

    # -----------------------------
    # RETENTION STRATEGIES
    # -----------------------------
    actions = map_rules_to_actions(top_reasons)

    improved_rate = max(churn_rate - 0.10, 0)

    # -----------------------------
    # CUSTOMER TABLE (TOP 10 RISK)
    # -----------------------------
    results = []

    for i in range(len(probs)):

        risk = "Low Risk"

        if probs[i] >= 0.33:
            risk = "High Risk"
        elif probs[i] >= 0.2:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        results.append({
            "customer_id": str(i+1),
            "probability": float(probs[i]),
            "risk": risk
        })

    # Sort by probability
    results = sorted(results, key=lambda x: x["probability"], reverse=True)

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {

        "total_customers": total_customers,
        "total_churn": total_churn,
        "churn_rate": float(churn_rate),

        "risk_distribution": risk_distribution,

        "top_reasons": top_reasons,

        "recommended_actions": actions,

        "new_probability_after_strategy": float(improved_rate),

        "customers": results[:10]   # top 10 risky customers

    }

    