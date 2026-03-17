import pandas as pd
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load model
# --------------------------------------------------
model = joblib.load("data/random_forest_model.pkl")

# --------------------------------------------------
# 2. Load test data
# --------------------------------------------------
X_test = pd.read_csv("data/X_test.csv")

# Ensure numeric (very important)
X_test = X_test.astype(float)

print("Model and test data loaded.")
print("X_test shape:", X_test.shape)
# --------------------------------------------------
# 3. Create SHAP TreeExplainer
# --------------------------------------------------
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# --------------------------------------------------
# 4. Handle ALL possible SHAP output formats safely
# --------------------------------------------------

if isinstance(shap_values, list):
    # Old format: list of arrays [class0, class1]
    shap_class1 = shap_values[1]

elif isinstance(shap_values, np.ndarray):

    if shap_values.ndim == 3:
        # New format: (samples, features, classes)
        shap_class1 = shap_values[:, :, 1]

    elif shap_values.ndim == 2:
        # Already correct
        shap_class1 = shap_values

    else:
        raise ValueError("Unexpected SHAP ndarray shape")

else:
    raise ValueError("Unexpected SHAP output format")

print("Final SHAP shape:", shap_class1.shape)
# Should now be: (1126, 37)

# --------------------------------------------------
# 5. GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------

mean_abs_shap = np.abs(shap_class1).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsSHAP": mean_abs_shap
}).sort_values(by="MeanAbsSHAP", ascending=False)

print("\nTop 10 Global Important Features:")
print(importance_df.head(10))

# --------------------------------------------------
# 6. Explain ONE customer
# --------------------------------------------------

customer_index = 0

customer_shap = shap_class1[customer_index]

customer_df = pd.DataFrame({
    "Feature": X_test.columns,
    "SHAP_Value": customer_shap
})

customer_df["Abs_SHAP"] = customer_df["SHAP_Value"].abs()
customer_df = customer_df.sort_values(by="Abs_SHAP", ascending=False)

print("\nTop 5 churn-driving factors for this customer:")
print(customer_df.head(5))

from retention_engine import (
    get_top_shap_reasons,
    map_rules_to_actions,
    generate_llm_strategy
)

API_KEY = "YOUR_OPENROUTER_API_KEY"

customer_index = 0

# Get top churn reasons
top_reasons = get_top_shap_reasons(
    shap_class1[customer_index],
    X_test.columns,
    top_n=3
)

print("\nTop SHAP churn reasons:")
print(top_reasons)

# Map rules
actions = map_rules_to_actions(top_reasons)

print("\nRule-Based Retention Actions:")
for act in actions:
    print("-", act)

# LLM Strategy
strategy = generate_llm_strategy(
    API_KEY,
    top_reasons,
    actions
)

print("\nAI-Generated Retention Strategy:\n")
print(strategy)
