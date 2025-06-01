import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Set page config FIRST
st.set_page_config(page_title="Churn Dashboard", layout="wide")

#  Load model, scaler, and column order
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("model_trained.pkl")



# -------- MAIN TITLE --------
st.markdown(
    "<h1 style='text-align: center; color: #4A7BF7;'>Customer Churn Dashboard</h1>",
    unsafe_allow_html=True
)

# Layout: Input Form (2/3 width) | Empty right column (no prediction here)
left_col, right_col = st.columns([2, 1])

# -------- INPUT FORM --------
with left_col:
    st.markdown("###  Customer Profile")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with c2:
        senior = st.radio("Senior Citizen", ["Yes", "No"], horizontal=True)
    with c3:
        partner = st.radio("Has Partner?", ["Yes", "No"], horizontal=True)
    with c4:
        dependents = st.radio("Has Dependents?", ["Yes", "No"], horizontal=True)

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.radio("Paperless Billing?", ["Yes", "No"], horizontal=True)
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    c5, c6, c7 = st.columns(3)
    with c5:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    with c6:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    with c7:
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2500.0)

    # Prepare input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'gender_Male': [1 if gender == "Male" else 0],
        'SeniorCitizen_Yes': [1 if senior == "Yes" else 0],
        'Partner_Yes': [1 if partner == "Yes" else 0],
        'Dependents_Yes': [1 if dependents == "Yes" else 0],
        'Contract_One year': [1 if contract == "One year" else 0],
        'Contract_Two year': [1 if contract == "Two year" else 0],
        'PaperlessBilling_Yes': [1 if paperless == "Yes" else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment == "Credit card (automatic)" else 0],
        'PaymentMethod_Electronic check': [1 if payment == "Electronic check" else 0],
        'PaymentMethod_Mailed check': [1 if payment == "Mailed check" else 0],
    }, columns=columns)

    input_data.fillna(0, inplace=True)
    input_data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_data[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

# -------- STORE PREDICTION (defer display) --------
pred = None
prob = None
if st.button("Predict Now"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

# -------- DISPLAY PREDICTION & GAUGE --------
if pred is not None:
    result_color = "red" if pred == 1 else "green"
    result_msg = "❌ High Risk of Churn" if pred == 1 else "✅ Low Risk of Churn"
    prob_display = prob if pred == 1 else 100 - prob

    st.markdown(
        f"<div style='background-color:{result_color}; padding: 20px; border-radius: 10px; text-align: center;'>"
        f"<h2 style='color:white;'>{result_msg}</h2>"
        f"<h3 style='color:white;'>Probability: {prob_display:.2f}%</h3>"
        "</div>", unsafe_allow_html=True
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': result_color},
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# -------- FOOTER --------
st.markdown("<hr style='margin-top: 20px;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Dashboard developed with ❤️ by <b>Ratnala Raja</b></p>", unsafe_allow_html=True)
