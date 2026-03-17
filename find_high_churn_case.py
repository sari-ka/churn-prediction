import pandas as pd
import joblib

# load trained model
model = joblib.load("data/random_forest_model.pkl")

# load dataset used for training
X = pd.read_csv("data/X_test.csv")

# predict probabilities
probs = model.predict_proba(X)[:,1]

# attach probabilities
X["churn_probability"] = probs

# show highest churn cases
top = X.sort_values("churn_probability", ascending=False).head(5)

print("\nTop High-Risk Customers:\n")
print(top)