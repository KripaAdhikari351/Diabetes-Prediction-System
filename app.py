"""
app.py

Streamlit front-end for the Diabetes Prediction System. Loads the model
trained by train_model.py and lets a user enter their own health details
to get an instant risk prediction.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="centered")

# ---------------------------------------------------------
# Load trained model, scaler, and expected feature order
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_regression_model.joblib")
    scaler = joblib.load("diabetes_scaler.joblib")
    columns = joblib.load("feature_columns.joblib")
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

st.title("🩺 Diabetes Prediction System")
st.write("Enter your health details below, then click **Predict** to see your estimated diabetes risk.")
st.caption("This is an educational tool trained on synthetic data — not a medical diagnosis.")

# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", value=100.0)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", value=80.0)
    serum_glucose = st.number_input("Serum Glucose (mg/dL)", value=110.0)

with col2:
    hba1c = st.number_input("HbA1c (%)", value=5.5)
    hdl = st.number_input("HDL Cholesterol (mg/dL)", value=50.0)
    ldl = st.number_input("LDL Cholesterol (mg/dL)", value=110.0)
    triglycerides = st.number_input("Triglycerides (mg/dL)", value=130.0)
    waist = st.number_input("Waist Circumference (cm)", value=90.0)
    creatinine = st.number_input("Creatinine (mg/dL)", value=0.9)

st.divider()
c1, c2, c3 = st.columns(3)
with c1:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
with c2:
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
with c3:
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])

# ---------------------------------------------------------
# Encode categorical inputs to match training data
# ---------------------------------------------------------
gender_male = 1 if gender == "Male" else 0
smoking_yes = 1 if smoking == "Yes" else 0
family_yes = 1 if family_history == "Yes" else 0
hypertension_yes = 1 if hypertension == "Yes" else 0

input_values = {
    "Age": age,
    "Pregnancies": pregnancies,
    "BMI": bmi,
    "FastingGlucose": fasting_glucose,
    "DiastolicBP": diastolic_bp,
    "HbA1c": hba1c,
    "SerumGlucose": serum_glucose,
    "HDL": hdl,
    "LDL": ldl,
    "Triglycerides": triglycerides,
    "WaistCircumference": waist,
    "Creatinine": creatinine,
    "Gender_Male": gender_male,
    "Smoking_Yes": smoking_yes,
    "FamilyHistory_Yes": family_yes,
    "Hypertension_Yes": hypertension_yes,
}

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict", type="primary"):
    patient = np.array([[input_values[col] for col in feature_columns]])
    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ There is a higher likelihood of diabetes based on these values.")
    else:
        st.success("✅ There is no significant indication of diabetes based on these values.")

    st.write(f"**Estimated probability of diabetes:** {probability * 100:.2f}%")
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("This tool does not replace professional medical advice — please consult a doctor for an actual diagnosis.")
