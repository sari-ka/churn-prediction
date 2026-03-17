import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --------------------------------------------------
# 1. Load feature-engineered data
# --------------------------------------------------
df = pd.read_csv("data/ecommerce_featured.csv")

# --------------------------------------------------
# 2. Select ONLY numeric features for VIF
# (categorical will be encoded later)
# --------------------------------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop('Churn')

X = df[numeric_cols]

# --------------------------------------------------
# 3. Compute VIF
# --------------------------------------------------
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

vif_data = vif_data.sort_values(by='VIF', ascending=False)

print("\nVIF Results (Multicollinearity Check):")
print(vif_data)
