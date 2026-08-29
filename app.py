import streamlit as st
import joblib
import numpy as np


# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ---------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()


# ---------------------------------------------------
# App title
# ---------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter the customer's details below to predict whether "
    "the customer is likely to churn."
)

st.divider()


# ---------------------------------------------------
# Input fields
# ---------------------------------------------------

age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=30,
    step=1
)

tenure = st.number_input(
    "Tenure",
    min_value=0,
    max_value=130,
    value=10,
    step=1
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🔮 Predict Churn", use_container_width=True):

    # Female = 1, Male = 0
    gender_selected = 1 if gender == "Female" else 0

    # Keep the exact feature order used during training
    X = np.array([
        age,
        gender_selected,
        tenure,
        monthly_charges
    ]).reshape(1, -1)

    # Scale input
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)[0]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Prediction: Customer is likely to CHURN")
    else:
        st.success("✅ Prediction: Customer is NOT likely to churn")
