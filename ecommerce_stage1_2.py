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
# 2. Load data
# --------------------------------------------------
X_test = pd.read_csv("data/X_test.csv")

# FORCE numeric
X_test = X_test.astype(float)

print("Model and numeric test data loaded.")
print("Shape of X_test:", X_test.shape)

# --------------------------------------------------
# 3. Use TreeExplainer (stable mode)
# --------------------------------------------------
explainer = shap.TreeExplainer(model)

# IMPORTANT:
# For sklearn RandomForestClassifier,
# shap_values returns a list:
# shap_values[0] → class 0
# shap_values[1] → class 1 (churn)

shap_values = explainer.shap_values(X_test)

# Select class 1 (churn)
shap_class1 = shap_values[1]

print("Shape of SHAP matrix:", shap_class1.shape)

# --------------------------------------------------
# 4. GLOBAL FEATURE IMPORTANCE (manual calculation)
# --------------------------------------------------
# Instead of summary_plot (causing shape mismatch),
# we compute mean absolute SHAP importance manually

mean_importance = np.abs(shap_class1).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Abs_SHAP": mean_importance
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

print("\nTop 10 Global Important Features:")
print(importance_df.head(10))

# --------------------------------------------------
# 5. Explain ONE customer safely
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

# --------------------------------------------------
# 6. Optional: Simple Bar Plot (stable)
# --------------------------------------------------
top_features = importance_df.head(10)

plt.figure(figsize=(8,5))
plt.barh(top_features["Feature"], top_features["Mean_Abs_SHAP"])
plt.gca().invert_yaxis()
plt.title("Top 10 Global SHAP Importance")
plt.xlabel("Mean |SHAP value|")
plt.show()
