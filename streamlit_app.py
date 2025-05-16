import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("📉 Customer Churn Prediction App")

model = joblib.load("logistic_model.pkl") 
columns = joblib.load("feature_columns.pkl")  

st.sidebar.header("📋 Enter Customer Details")

monthly_charges = st.sidebar.slider("Monthly Charges", 20, 120, 70)
tenure = st.sidebar.slider("Tenure Months", 0, 72, 12)
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

input_dict = {
    'Monthly Charges': monthly_charges,
    'Tenure Months': tenure,
    'Internet Service': internet_service,
    'Contract': contract,
    'Paperless Billing': paperless
}

user_input = pd.DataFrame([input_dict])

user_input_encoded = pd.get_dummies(user_input)
user_input_encoded = user_input_encoded.reindex(columns=columns, fill_value=0)

prediction = model.predict(user_input_encoded)[0]
probability = model.predict_proba(user_input_encoded)[0][1]

st.subheader("Prediction Result")
st.write("🔍 Likely to Churn?" , "Yes" if prediction == 1 else "No")
st.write(f"💡 Probability of Churn: **{probability:.2f}**")

import shap
import matplotlib.pyplot as plt

st.subheader("🔍 Explanation: Why this prediction?")

background = pd.DataFrame(np.zeros((100, len(user_input_encoded.columns))), columns=user_input_encoded.columns)

explainer = shap.Explainer(model, background)

shap_values = explainer(user_input_encoded)

shap.plots.force(shap_values[0], matplotlib=True, show=False)
st.pyplot(bbox_inches='tight')