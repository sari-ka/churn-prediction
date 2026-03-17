import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Load feature-selected data
# --------------------------------------------------
df = pd.read_csv("data/ecommerce_selected.csv")

print("Input shape:", df.shape)

# --------------------------------------------------
# 2. Separate target and features
# --------------------------------------------------
X = df.drop(columns=['Churn'])
y = df['Churn']

# --------------------------------------------------
# 3. Identify categorical and numerical columns
# --------------------------------------------------
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

print("\nCategorical columns:")
print(cat_cols.tolist())

print("\nNumerical columns:")
print(num_cols.tolist())

# --------------------------------------------------
# 4. One-Hot Encode categorical variables
# --------------------------------------------------
X_encoded = pd.get_dummies(
    X,
    columns=cat_cols,
    drop_first=True  # avoid dummy variable trap
)

print("\nShape after encoding:", X_encoded.shape)

# --------------------------------------------------
# 5. Train-test split (STRATIFIED)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------------------------------------
# 6. Feature scaling (numerical only)
# --------------------------------------------------
scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# --------------------------------------------------
# 7. Save prepared datasets
# --------------------------------------------------
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\n✅ Stage 6 (Data Preparation) completed successfully.")
print("Saved: X_train, X_test, y_train, y_test")
