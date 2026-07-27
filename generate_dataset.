"""
generate_dataset.py

Creates a synthetic (but medically-plausible) patient dataset for training
the diabetes prediction model. Each row is a patient; 'Outcome' is 1 if the
patient has diabetes, 0 otherwise.

Run this once before train_model.py if diabetes_dataset.csv doesn't exist yet.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

age = np.random.randint(18, 85, N)
gender_male = np.random.randint(0, 2, N)
pregnancies = np.where(gender_male == 1, 0, np.random.randint(0, 6, N))
bmi = np.round(np.random.normal(27, 6, N).clip(15, 55), 1)
fasting_glucose = np.round(np.random.normal(100, 25, N).clip(60, 250), 1)
diastolic_bp = np.round(np.random.normal(78, 10, N).clip(50, 120), 1)
hba1c = np.round(np.random.normal(5.6, 1.1, N).clip(3.5, 12), 1)
serum_glucose = np.round(fasting_glucose + np.random.normal(15, 10, N), 1)
hdl = np.round(np.random.normal(50, 12, N).clip(20, 100), 1)
ldl = np.round(np.random.normal(110, 25, N).clip(40, 220), 1)
triglycerides = np.round(np.random.normal(130, 45, N).clip(40, 400), 1)
waist = np.round(np.random.normal(90, 13, N).clip(60, 150), 1)
creatinine = np.round(np.random.normal(0.9, 0.2, N).clip(0.4, 2.0), 2)
smoking = np.random.binomial(1, 0.22, N)
family_history = np.random.binomial(1, 0.3, N)
hypertension = np.random.binomial(1, 0.28, N)

# Build a risk score from the features that plausibly drive diabetes risk,
# then convert to a probability so the labels aren't random noise.
risk = (
    0.05 * (age - 45)
    + 0.16 * (bmi - 27)
    + 0.09 * (fasting_glucose - 100)
    + 1.8 * (hba1c - 5.6)
    + 0.03 * (triglycerides - 130)
    - 0.05 * (hdl - 50)
    + 0.06 * (waist - 90)
    + 1.4 * family_history
    + 0.9 * hypertension
    + 0.7 * smoking
)
prob = 1 / (1 + np.exp(-0.5 * risk))
outcome = np.random.binomial(1, prob)

df = pd.DataFrame({
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
    "Smoking_Yes": smoking,
    "FamilyHistory_Yes": family_history,
    "Hypertension_Yes": hypertension,
    "Outcome": outcome,
})

df.to_csv("diabetes_dataset.csv", index=False)
print(f"Saved diabetes_dataset.csv with {len(df)} rows.")
print(f"Diabetes-positive rate: {df['Outcome'].mean():.1%}")
