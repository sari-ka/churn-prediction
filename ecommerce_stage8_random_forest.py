import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# --------------------------------------------------
# 1. Load prepared data
# --------------------------------------------------

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# --------------------------------------------------
# 2. Create Pipeline (Scaling + Random Forest)
# --------------------------------------------------

pipeline = Pipeline([

    ("scaler", StandardScaler()),

    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))

])

# --------------------------------------------------
# 3. Train Model
# --------------------------------------------------

pipeline.fit(X_train, y_train)

# --------------------------------------------------
# 4. Predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:,1]

# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:", roc_auc)

# --------------------------------------------------
# 6. ROC Curve
# --------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()

plt.plot(fpr, tpr, label=f"Random Forest ROC-AUC = {roc_auc:.3f}")

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.legend()

plt.show()

# --------------------------------------------------
# 7. Save Model Pipeline
# --------------------------------------------------

joblib.dump(pipeline, "data/random_forest_model.pkl")

print("\nModel Pipeline saved as data/random_forest_model.pkl")