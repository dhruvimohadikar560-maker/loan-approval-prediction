import streamlit as st
import pickle
import numpy as np
import os

# Get current folder of app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.pkl")

# Load model
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Page Config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("Enter applicant details to predict loan approval status.")

# Input Fields
no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    step=1
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["No", "Yes"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=1
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0
)

# Encoding
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# Prediction Button
if st.button("Predict Loan Status"):

    input_data = np.array([[
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]])

    try:
        prediction = model.predict(input_data)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error during prediction: {e}")