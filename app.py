import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺")

model = joblib.load("logistic_regression_model.joblib")
scaler = joblib.load("diabetes_scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")

st.title("🩺 Diabetes Prediction System")
st.write("Enter your details below and click Predict.")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", ["Female", "Male"])
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
fasting_glucose = st.number_input("Fasting Glucose", value=100.0)
diastolic_bp = st.number_input("Diastolic Blood Pressure", value=80.0)
serum_glucose = st.number_input("Serum Glucose", value=110.0)
hba1c = st.number_input("HbA1c", value=5.5)
hdl = st.number_input("HDL Cholesterol", value=50.0)
ldl = st.number_input("LDL Cholesterol", value=110.0)
triglycerides = st.number_input("Triglycerides", value=130.0)
waist = st.number_input("Waist Circumference", value=90.0)
creatinine = st.number_input("Creatinine", value=0.9)
smoking = st.selectbox("Smoking", ["No", "Yes"])
family_history = st.selectbox("Family History", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])

# convert yes/no stuff to 0/1
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

if st.button("Predict"):
    patient = np.array([[input_values[col] for col in feature_columns]])
    patient_scaled = scaler.transform(patient)

    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ There is a chance of having diabetes.")
    else:
        st.success("✅ There is no significant chance of having diabetes.")

    st.write(f"Probability of diabetes: {probability*100:.2f}%")
