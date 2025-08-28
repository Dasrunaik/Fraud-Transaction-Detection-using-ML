import streamlit as st
import pandas as pd
import joblib

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# Expected feature columns (must match training!)
expected_cols = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud"
]

# Streamlit UI
st.title("💳 Fraud Detection App")
st.write("Enter transaction details to predict if it's Fraudulent or Legit.")

# Input form
with st.form("fraud_form"):
    step = st.number_input("Step (time in hours)", min_value=0, max_value=744, value=1)
    type_ = st.selectbox("Transaction Type", ["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)
    newbalanceOrg = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=2000.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud?", [0, 1])

    submitted = st.form_submit_button("Predict Fraud")

# When user clicks "Predict Fraud"
if submitted:
    # Create input DataFrame
    data = pd.DataFrame([{
        "step": step,
        "type": type_,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrg": newbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFlaggedFraud": isFlaggedFraud
    }])

    # Ensure correct column order
    data = data.reindex(columns=expected_cols)

    # Make prediction
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    # Show result
    if pred == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {prob:.2f})")
