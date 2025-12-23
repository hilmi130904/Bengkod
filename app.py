import streamlit as st
import pandas as pd
import joblib

st.title("Telco Churn Prediction")

model = joblib.load("best_churn_model.joblib")

st.write("Upload data atau isi input (contoh sederhana).")
