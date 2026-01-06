# ====== app.py ======
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📉 Prediksi Churn Pelanggan (Telco)")
st.write("Masukkan data pelanggan, lalu klik **Prediksi**.")

YES_NO = ["Yes", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]
CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]

@st.cache_resource
def load_model():
    return joblib.load("best_churn_model.joblib")

model = load_model()

def _get_classes(m):
    # model bisa pipeline / estimator biasa
    if hasattr(m, "classes_"):
        return list(m.classes_)
    if hasattr(m, "named_steps"):
        last_step = list(m.named_steps.values())[-1]
        if hasattr(last_step, "classes_"):
            return list(last_step.classes_)
    return None

def _positive_label(classes):
    """
    Tentukan label positif churn (Yes) dengan aman.
    """
    if not classes:
        return None
    # kalau ada label string "Yes", pakai itu
    for lab in classes:
        if str(lab).lower() == "yes":
            return lab
    # kalau ada angka 1, pakai 1
    if 1 in classes:
        return 1
    # fallback: ambil kelas terakhir (umum pada sklearn)
    return classes[-1]

def _is_churn_label(pred_label):
    """
    Deteksi churn dari label prediksi, aman untuk:
    - 'Yes'/'No'
    - 1/0
    """
    s = str(pred_label).strip().lower()
    return (s == "yes") or (s == "1") or (s == "true")

with st.form("churn_form"):
    customer_id = st.text_input(
        "Customer ID (wajib diisi)",
        placeholder="Contoh: 7590-VHVEG"
    )

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
        Partner = st.selectbox("Partner", YES_NO)
        Dependents = st.selectbox("Dependents", YES_NO)
        tenure = st.number_input("tenure (bulan)", min_value=0, max_value=100, value=12)

        PhoneService = st.selectbox("PhoneService", YES_NO)

        # ✅ MultipleLines tergantung PhoneService
        if PhoneService == "No":
            MultipleLines = st.selectbox(
                "MultipleLines",
                ["No phone service"],
                disabled=True
            )
        else:
            MultipleLines = st.selectbox("MultipleLines", ["Yes", "No"])

        InternetService = st.selectbox("InternetService", INTERNET_SERVICES)

        # ✅ Fitur internet tergantung InternetService
        if InternetService == "No":
            OnlineSecurity = st.selectbox("OnlineSecurity", ["No internet service"], disabled=True)
            OnlineBackup = st.selectbox("OnlineBackup", ["No internet service"], disabled=True)
        else:
            OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No"])
            OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No"])

    with col2:
        # ✅ Fitur internet tergantung InternetService
        if InternetService == "No":
            DeviceProtection = st.selectbox("DeviceProtection", ["No internet service"], disabled=True)
            TechSupport = st.selectbox("TechSupport", ["No internet service"], disabled=True)
            StreamingTV = st.selectbox("StreamingTV", ["No internet service"], disabled=True)
            StreamingMovies = st.selectbox("StreamingMovies", ["No internet service"], disabled=True)
        else:
            DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No"])
            TechSupport = st.selectbox("TechSupport", ["Yes", "No"])
            StreamingTV = st.selectbox("StreamingTV", ["Yes", "No"])
            StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No"])

        Contract = st.selectbox("Contract", CONTRACTS)
        PaperlessBilling = st.selectbox("PaperlessBilling", YES_NO)
        PaymentMethod = st.selectbox("PaymentMethod", PAYMENT_METHODS)

        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, value=70.0, format="%.2f")
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=20000.0, value=1000.0, format="%.2f")

    submitted = st.form_submit_button("Prediksi")

if submitted:
    if customer_id.strip() == "":
        st.error("Customer ID wajib diisi!")
        st.stop()

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

    st.caption(f"Customer ID: **{customer_id.strip()}**")

    # ✅ Tampilkan ringkasan input (biar output sesuai input yang masuk ke model)
    st.subheader("Ringkasan input")
    st.dataframe(input_df, use_container_width=True)

    # ===== Prediksi =====
    pred_label = model.predict(input_df)[0]

    classes = _get_classes(model)
    pos_lab = _positive_label(classes)

    churn_proba = None
    if hasattr(model, "predict_proba") and classes is not None and pos_lab is not None:
        proba_all = model.predict_proba(input_df)[0]
        pos_idx = list(classes).index(pos_lab)
        churn_proba = float(proba_all[pos_idx])

    # ✅ Output label aman (Yes/No atau 1/0)
    if _is_churn_label(pred_label):
        st.error("⚠️ Prediksi: **CHURN (Yes)**")
    else:
        st.success("✅ Prediksi: **TIDAK CHURN (No)**")

    if churn_proba is not None:
        st.write(f"Probabilitas churn (Yes): **{churn_proba:.2f}**")
