# 🩺 Diabetes Prediction System

A simple machine learning web app that predicts diabetes risk from patient
health metrics (age, BMI, glucose, blood pressure, cholesterol, etc.) using
logistic regression, served through a Streamlit interface.

## Project structure

```
diabetes-prediction-system/
├── generate_dataset.py   # creates the synthetic training dataset
├── diabetes_dataset.csv  # training data 
├── train_model.py        # trains & saves the logistic regression model
├── app.py                # Streamlit app for live predictions
├── logistic_regression_model.joblib  # trained model 
├── diabetes_scaler.joblib            # feature scaler 
├── feature_columns.joblib            # expected feature order 
├── requirements.txt
└── README.md
```

## How it works

1. **`generate_dataset.py`** builds a synthetic dataset of patients with
   realistic health metrics and a diabetes `Outcome` label.
2. **`train_model.py`** loads the dataset, scales the features, trains a
   `LogisticRegression` classifier, prints accuracy/precision/recall, and
   saves the model + scaler to disk.
3. **`app.py`** loads those saved files and gives you a form to enter your
   own numbers, then shows a prediction and probability.

## Running it locally

```bash
pip install -r requirements.txt
python generate_dataset.py   # only needed once, to create the dataset
python train_model.py        # trains and saves the model
streamlit run app.py         # launches the web app in your browser
```

