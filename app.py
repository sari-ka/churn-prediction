import streamlit as st
import requests
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Customer Churn Intelligence Dashboard")

API_URL = "http://127.0.0.1:8000/predict"

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 36, 3)
    order_count = st.number_input("Order Count", 0, 100, 5)
    days_last = st.number_input("Days Since Last Order", 0, 30, 3)

with col2:
    cashback = st.number_input("Cashback Amount", 0, 10000, 500)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    complain = st.selectbox("Complaint", [0,1])

with col3:
    coupon = st.selectbox("Coupon Used", [0,1])
    city = st.selectbox("City Tier", [1,2,3])
    device = st.selectbox("Preferred Login Device", ["mobile","computer"])
    payment = st.selectbox(
    "Payment Mode",
    ["credit card","debit card","upi","cod","e wallet"]
    )

data = {
"Tenure": tenure,
"OrderCount": order_count,
"DaySinceLastOrder": days_last,
"CashbackAmount": cashback,
"SatisfactionScore": satisfaction,
"Complain": complain,
"CouponUsed": coupon,
"CityTier": city,
"PreferredLoginDevice": device,
"PreferredPaymentMode": payment
}

if st.button("Predict Churn"):

    response = requests.post(API_URL, json=data)
    result = response.json()

    prob = result["churn_probability"]
    prob_percent = round(prob*100,2)

    # UPDATED RISK THRESHOLDS
    if prob_percent < 20:
        risk = "Low"
    elif prob_percent < 40:
        risk = "Medium"
    else:
        risk = "High"

    st.markdown("---")

    # 1️⃣ CHURN PROBABILITY
    st.subheader("Churn Probability")

    st.metric("Probability", f"{prob_percent}%")
    st.write("Risk Level:", risk)

    # 2️⃣ TOP CHURN DRIVERS
    st.subheader("Top Churn Drivers")

    st.caption("X-axis shows SHAP impact (how strongly each factor pushes churn risk).")

    features = [r[0] for r in result["top_reasons"]]
    values = [abs(r[1]) for r in result["top_reasons"]]

    fig, ax = plt.subplots(figsize=(4,2))

    ax.barh(features, values)

    ax.set_xlabel("SHAP Impact")

    st.pyplot(fig)

    # 3️⃣ RETENTION TECHNIQUES
    st.subheader("Retention Techniques")

    for a in result["recommended_actions"]:
        st.write("-", a)

    # 4️⃣ RETENTION IMPACT
    st.subheader("Retention Impact")

    before = prob
    after = result["new_probability_after_strategy"]

    fig2, ax2 = plt.subplots(figsize=(4,2))

    ax2.bar(["Before Strategy","After Strategy"], [before, after])

    ax2.set_ylabel("Churn Probability")

    st.pyplot(fig2)
