import pandas as pd

# --------------------------------------------------
# 1. Load feature-engineered data
# --------------------------------------------------
df = pd.read_csv("data/ecommerce_featured.csv")

print("Initial shape:", df.shape)

# --------------------------------------------------
# 2. Drop highly collinear / weak features
# (Based on VIF + business reasoning)
# --------------------------------------------------
features_to_drop = [
    'OrderCount',
    'HourSpendOnApp',
    'DaySinceLastOrder',
    'CashbackAmount',
    'OrderAmountHikeFromlastYear',
    'NumberOfDeviceRegistered'
]

df_selected = df.drop(columns=features_to_drop)

print("Dropped features:", features_to_drop)
print("Shape after feature selection:", df_selected.shape)

# --------------------------------------------------
# 3. Sanity check
# --------------------------------------------------
print("\nRemaining columns:")
print(df_selected.columns.tolist())

print("\nMissing values check:")
print(df_selected.isnull().sum())

# --------------------------------------------------
# 4. Save final feature-selected dataset
# --------------------------------------------------
df_selected.to_csv("data/ecommerce_selected.csv", index=False)

print("\n✅ Stage 5 (Feature Selection) completed successfully.")
print("Final dataset saved to: data/ecommerce_selected.csv")
