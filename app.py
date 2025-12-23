# ====== app.py ======
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("📉 Prediksi Churn Pelanggan (Telco)")

@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.joblib")

model = load_model()

st.write("Masukkan data pelanggan, lalu klik **Prediksi**.")

# Opsi umum dataset Telco (aman untuk dropdown)
YES_NO = ["Yes", "No"]
YES_NO_NOINT = ["Yes", "No", "No internet service"]
YES_NO_NOPHONE = ["Yes", "No", "No phone service"]

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
        Partner = st.selectbox("Partner", YES_NO)
        Dependents = st.selectbox("Dependents", YES_NO)
        tenure = st.number_input("tenure (bulan)", min_value=0, max_value=100, value=12)

        PhoneService = st.selectbox("PhoneService", YES_NO)
        MultipleLines = st.selectbox("MultipleLines", YES_NO_NOPHONE)

        InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("OnlineSecurity", YES_NO_NOINT)
        OnlineBackup = st.selectbox("OnlineBackup", YES_NO_NOINT)

    with col2:
        DeviceProtection = st.selectbox("DeviceProtection", YES_NO_NOINT)
        TechSupport = st.selectbox("TechSupport", YES_NO_NOINT)
        StreamingTV = st.selectbox("StreamingTV", YES_NO_NOINT)
        StreamingMovies = st.selectbox("StreamingMovies", YES_NO_NOINT)

        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("PaperlessBilling", YES_NO)
        PaymentMethod = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0)
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=1000.0)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    if pred == 1:
        st.error("⚠️ Prediksi: CHURN (Yes)")
    else:
        st.success("✅ Prediksi: TIDAK CHURN (No)")

    if proba is not None:
        st.write(f"Probabilitas churn: **{proba:.2f}**")
