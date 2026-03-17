import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load EDA-ready data
# --------------------------------------------------
df = pd.read_csv("data/ecommerce_eda_ready.csv")

print("Input shape:", df.shape)

# --------------------------------------------------
# 2. TENURE-BASED FEATURES
# (Strong churn driver from EDA)
# --------------------------------------------------

# Tenure buckets
df['TenureBucket'] = pd.cut(
    df['Tenure'],
    bins=[-1, 3, 6, 12, 24, 100],
    labels=['0-3', '4-6', '7-12', '13-24', '24+']
)

# New customer flag
df['IsNewCustomer'] = (df['Tenure'] <= 3).astype(int)

# --------------------------------------------------
# 3. RECENCY & ACTIVITY FEATURES
# --------------------------------------------------

# Inactive customer flag
df['IsInactive'] = (df['DaySinceLastOrder'] > 7).astype(int)

# Engagement score (simple but powerful)
df['EngagementScore'] = (
    df['OrderCount'] +
    df['HourSpendOnApp'] -
    df['DaySinceLastOrder']
)

# --------------------------------------------------
# 4. MONETARY / VALUE FEATURES
# (Proxy for "charges")
# --------------------------------------------------

# High value customer
df['IsHighValueCustomer'] = (
    (df['CashbackAmount'] > df['CashbackAmount'].median()) &
    (df['OrderCount'] > df['OrderCount'].median())
).astype(int)

# Order frequency
df['OrdersPerTenure'] = df['OrderCount'] / (df['Tenure'] + 1)

# --------------------------------------------------
# 5. COMPLAINT & SATISFACTION FEATURES
# --------------------------------------------------

# Complaint risk flag
df['ComplaintRisk'] = (
    (df['Complain'] == 1) &
    (df['SatisfactionScore'] <= 3)
).astype(int)

# Satisfaction bucket
df['SatisfactionBucket'] = pd.cut(
    df['SatisfactionScore'],
    bins=[0, 2, 3, 5],
    labels=['Low', 'Medium', 'High']
)

# --------------------------------------------------
# 6. DEVICE & PAYMENT RISK FLAGS
# (Based on EDA churn rates)
# --------------------------------------------------

df['HighRiskDevice'] = df['PreferredLoginDevice'].isin(
    ['phone']
).astype(int)

df['HighRiskPaymentMode'] = df['PreferredPaymentMode'].isin(
    ['cod', 'e wallet']
).astype(int)

# --------------------------------------------------
# 7. FINAL CHECK
# --------------------------------------------------

print("\nNew features added:")
new_features = set(df.columns) - set(pd.read_csv("data/ecommerce_eda_ready.csv").columns)
print(sorted(new_features))

print("\nFinal shape after feature engineering:", df.shape)
print("\nMissing values after feature engineering:")
print(df.isnull().sum())

# --------------------------------------------------
# 8. SAVE FEATURE-ENGINEERED DATA
# --------------------------------------------------

df.to_csv("data/ecommerce_featured.csv", index=False)

print("\n✅ Stage 4 (Feature Engineering) completed successfully.")
print("Feature-engineered data saved to: data/ecommerce_featured.csv")
