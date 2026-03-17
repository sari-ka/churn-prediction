import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------
# 1. Load cleaned data
# --------------------------------------------------
df = pd.read_csv("data/ecommerce_cleaned.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. TARGET ANALYSIS — Churn Distribution
# --------------------------------------------------
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print("\nChurn rate:")
print(df['Churn'].value_counts(normalize=True))

df['Churn'].value_counts().plot(kind='bar')
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Customer Count")
plt.show()

# --------------------------------------------------
# 3. NUMERICAL FEATURES vs CHURN
# --------------------------------------------------
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Churn')

print("\nNumerical feature means grouped by Churn:")
print(df.groupby('Churn')[num_cols].mean())

# Boxplots for key churn-related numerical features
key_num_features = [
    'Tenure',
    'DaySinceLastOrder',
    'OrderCount',
    'CashbackAmount',
    'HourSpendOnApp'
]

for col in key_num_features:
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f"{col} vs Churn")
    plt.show()

# --------------------------------------------------
# 4. CATEGORICAL FEATURES vs CHURN
# (Used instead of contract type — not present in data)
# --------------------------------------------------
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    print(f"\nChurn rate by {col}:")
    churn_rate = df.groupby(col)['Churn'].mean().sort_values(ascending=False)
    print(churn_rate)

    churn_rate.plot(kind='bar')
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.show()

# --------------------------------------------------
# 5. CORRELATION ANALYSIS (Numerical only)
# --------------------------------------------------
corr_df = df[num_cols.tolist() + ['Churn']].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

print("\nCorrelation with Churn:")
print(corr_df['Churn'].sort_values(ascending=False))

# --------------------------------------------------
# 6. SAVE EDA-READY DATASET
# --------------------------------------------------
df.to_csv("data/ecommerce_eda_ready.csv", index=False)

print("\n✅ Stage 3 (EDA) completed successfully.")
print("EDA-ready data saved to: data/ecommerce_eda_ready.csv")
