# 🚀 Customer Churn Prediction & Retention Intelligence System

An end-to-end machine learning system that predicts customer churn and generates actionable retention strategies using explainable AI.

---

## 📌 Overview

This project goes beyond basic churn prediction by combining:

- 🔮 Predictive Modeling (ML)
- 📊 Explainable AI (SHAP)
- 💡 Automated Retention Recommendations
- 📈 Interactive Dashboard for decision-making

👉 Goal: Help businesses proactively identify high-risk customers and reduce churn.

---

## 🧠 Key Features

### 🔹 1. Churn Prediction Engine
- Built classification models (Random Forest / XGBoost)
- Achieved **85%+ accuracy**
- Predicts churn probability in real-time

---

### 🔹 2. Explainable AI (SHAP)
- Provides feature-level impact on predictions
- Identifies key churn drivers like:
  - Tenure
  - Satisfaction score
  - Complaints
  - Purchase behavior

---

### 🔹 3. Retention Intelligence System
- Converts model insights into **actionable strategies**
- Example:
  - Offer targeted discounts
  - Improve support response
  - Run engagement campaigns

---

### 🔹 4. Retention Impact Simulation
- Visualizes churn before vs after strategy
- Helps evaluate effectiveness of interventions

---

### 🔹 5. Interactive Dashboard
- Built using React
- Includes:
  - 📊 Churn probability
  - 📈 Feature importance chart
  - 📉 Retention impact
  - 💡 Recommendations

---

### 🔹 6. Batch Analysis
- Processes 1000+ customer records
- Identifies high-risk segments
- Enables large-scale decision making

---

## 🏗️ Tech Stack

### Backend
- FastAPI
- Python
- Scikit-learn
- SHAP

### Frontend
- React.js
- Chart.js / Recharts

### Data
- E-commerce customer dataset

---

## ⚙️ System Architecture

User Input → FastAPI → ML Model → SHAP → Retention Engine → Dashboard

---

## 📸 Sample Outputs

### 🔹 Single Customer Churn Prediction Output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7dc97759-2b4d-42be-93f6-b968e1217bcc" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/46c87a0d-1ffe-480e-b113-6efc5a3ae939" />

### 🔹 Batch of  Customers Churn Prediction Output
<img width="701" height="853" alt="image" src="https://github.com/user-attachments/assets/8ca6143a-2c5b-4a1a-87cd-dfc56df69d21" />
<img width="710" height="403" alt="image" src="https://github.com/user-attachments/assets/be4a34d2-c68f-43a2-b9ec-5ba9f968f8c6" />


---

## 🚀 How to Run

### 1. Clone repo
```bash
git clone https://github.com/sari-ka/churn-prediction.git
cd churn-prediction
```

### 2. Setup backend
```bash
cd backend
pip install -r requirements.txt
uvicorn backend_api:app --reload
```

### 3. Setup frontend
```bash
cd churn-dashboard
npm install
npm run dev
```



