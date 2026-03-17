import requests
import json

API_KEY = "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"

url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",   # required
    "X-Title": "Churn Retention Project"  # required
}

data = {
    "model": "mistralai/mistral-7b-instruct",
    "messages": [
        {
            "role": "user",
            "content": "Explain why customer engagement matters for retention in simple business language."
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))

result = response.json()

print(result["choices"][0]["message"]["content"])
