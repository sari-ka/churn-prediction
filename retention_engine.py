import requests
import json

def generate_llm_strategy(api_key, churn_reasons, actions):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Churn Retention Project"
    }

    reasons_text = ", ".join([r[0] for r in churn_reasons])
    actions_text = "; ".join(actions)

    prompt = f"""
    A customer has high churn risk.

    Key churn drivers:
    {reasons_text}

    Proposed retention actions:
    {actions_text}

    Convert this into a professional business retention strategy paragraph.
    Keep it concise and management-friendly.
    """

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    result = response.json()

    # ---- SAFE HANDLING ----
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        print("\n⚠️ LLM API Error Response:\n", result)
        return "LLM response unavailable due to API limit. Rule-based strategy applied."


def get_top_shap_reasons(shap_values_row, feature_names, top_n=3):
    feature_impact = list(zip(feature_names, shap_values_row))
    churn_drivers = [(f, v) for f, v in feature_impact if v > 0]
    churn_drivers.sort(key=lambda x: abs(x[1]), reverse=True)
    return churn_drivers[:top_n]


def map_rules_to_actions(top_features):

    actions = []

    for feature, value in top_features:

        if feature == "Tenure":
            actions.append("Provide onboarding loyalty discount")

        elif feature == "EngagementScore":
            actions.append("Send personalized re-engagement emails and app notifications")

        elif feature == "Complain":
            actions.append("Initiate priority customer support follow-up")

        elif feature == "IsNewCustomer":
            actions.append("Offer welcome coupon bundle and onboarding guidance")

        elif feature == "OrdersPerTenure":
            actions.append("Recommend personalized product bundles based on past purchases")

        elif "PreferedOrderCat" in feature:
            actions.append("Provide targeted discounts for preferred product category")

        elif feature == "SatisfactionScore":
            actions.append("Launch satisfaction recovery survey and service improvement outreach")

        elif feature == "CityTier":
            actions.append("Offer region-specific promotional campaigns")

        elif feature == "CouponUsed":
            actions.append("Provide limited-time exclusive coupon incentives")

        else:
            actions.append("Offer personalized retention incentive")

    return list(set(actions))

if __name__ == "__main__":

    print("Retention Engine Running...\n")

    # Example SHAP values (dummy test)
    shap_values_row = [0.2, -0.1, 0.35, 0.15, 0.05]

    feature_names = [
        "Tenure",
        "EngagementScore",
        "Complain",
        "OrdersPerTenure",
        "CouponUsed"
    ]

    # Step 1: get top churn drivers
    top_features = get_top_shap_reasons(shap_values_row, feature_names)

    print("Top churn drivers:")
    print(top_features)

    # Step 2: map actions
    actions = map_rules_to_actions(top_features)

    print("\nRecommended actions:")
    print(actions)

    # Step 3: LLM strategy
    api_key = "YOUR_OPENROUTER_API_KEY"

    strategy = generate_llm_strategy(api_key, top_features, actions)

    print("\nGenerated retention strategy:\n")
    print(strategy)