import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

# 2. Load ML Models safely
@st.cache_resource
def load_models():
    transformer = joblib.load("loan_transformer.pkl")
    model = joblib.load("Final_Model.pkl")
    encoder = joblib.load("Lable_Encoder.pkl")
    return transformer, model, encoder
    
# Load them directly so they are always available globally
transformer, model, encoder = load_models()

# 3. Core Prediction Function
def predict_loan(dependents, education, employment, income_amount, loan_amount, loan_term, cibil_score, residential_asset, commercial_asset, luxury_assets, bank_assets):
    user_data = pd.DataFrame({
        'no_of_dependents': [dependents],
        'education': [" " + education],
        'self_employed': [" " + employment],
        'income_annum': [income_amount],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_asset],
        'commercial_assets_value': [commercial_asset],
        'luxury_assets_value': [luxury_assets],
        'bank_asset_value': [bank_assets]
    })
    user_data_transform = transformer.transform(user_data)
    user_prediction = model.predict(user_data_transform)
    result = encoder.inverse_transform(user_prediction)
    return result[0].strip()

# 4. Streamlit UI Layout
st.title("🏦 Loan Approval Prediction")
st.caption("Developed by Swastika Barui | MSc Statistics, University of Calcutta")

st.warning(
    "⚠️ **Educational ML Project:** This application is for demonstration purposes only "
    "and is not connected to any bank or credit bureau. Please do not enter sensitive financial information."
)

# Multi-page system using Session State
if 'page' not in st.session_state:
    st.session_state.page = 1

# -------- PAGE 1: Applicant Details --------
if st.session_state.page == 1:
    st.subheader("👤 Applicant Details")
    name = st.text_input("Name", placeholder="Enter your name")
    age = st.number_input("Age", min_value=18, max_value=100, step=1, value=25)
    
    if st.button("Next ➡️"):
        if name.strip() == "":
            st.error("Please enter your name to proceed.")
        else:
            st.session_state.page = 2
            st.rerun()

# -------- PAGE 2: Loan Information --------
elif st.session_state.page == 2:
    st.subheader("📊 Loan Information")
    
    col1, col2 = st.columns(2)
    with col1:
        dependents = st.number_input("Financially dependent people count:", min_value=0, step=1, value=0)
        education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        employment = st.selectbox("Are you self-employed?", options=["Yes", "No"])
        income_amount = st.number_input("Annual income (in ₹)", min_value=0, value=500000)
        loan_amount = st.number_input("Desired loan amount (in ₹)", min_value=0, value=200000)
    
    with col2:
        loan_term = st.number_input("Loan term (in years)", min_value=1, step=1, value=5)
        cibil_score = st.number_input("CIBIL score (0 to 900)", min_value=0, max_value=900, step=1, value=750)
        residential_asset = st.number_input("Value of residential assets", min_value=0, value=0)
        commercial_asset = st.number_input("Value of commercial assets", min_value=0, value=0)
        luxury_assets = st.number_input("Value of luxury assets", min_value=0, value=0)
        bank_assets = st.number_input("Value of bank assets", min_value=0, value=0)

    st.info("💡 Enter your CIBIL score if you know it. This tool is not connected to a live credit bureau.")
    
    col_back, col_pred = st.columns([1, 4])
    with col_back:
        if st.button("⬅️ Back"):
            st.session_state.page = 1
            st.rerun()
            
    with col_pred:
        if st.button("🚀 Predict Loan Status", type="primary"):
            status = predict_loan(
                dependents, education, employment, income_amount, loan_amount, 
                loan_term, cibil_score, residential_asset, commercial_asset, 
                luxury_assets, bank_assets
            )
            
            st.markdown("---")
            if status == "Approved":
                st.success("### ✅ Loan Approved")
            else:
                st.error("### ❌ Loan Rejected")
