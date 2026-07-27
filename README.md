# 🩺 Diabetes Prediction System

A simple machine learning web app that predicts diabetes risk from patient
health metrics (age, BMI, glucose, blood pressure, cholesterol, etc.) using
logistic regression, served through a Streamlit interface.

## Project structure

```
diabetes-prediction-system/
├── generate_dataset.py   # creates the synthetic training dataset
├── diabetes_dataset.csv  # training data (generated)
├── train_model.py        # trains & saves the logistic regression model
├── app.py                # Streamlit app for live predictions
├── logistic_regression_model.joblib  # trained model (generated)
├── diabetes_scaler.joblib            # feature scaler (generated)
├── feature_columns.joblib            # expected feature order (generated)
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

## Pushing this to your own GitHub

I can't push to your GitHub account directly (I don't have your
credentials), but here's the fastest way to do it yourself:

1. Go to https://github.com/new and create a new repository
   (e.g. `Diabetes-Prediction-System`) — don't initialize it with a README.
2. In a terminal, `cd` into this project folder, then run:

```bash
git init
git add .
git commit -m "Initial commit: diabetes prediction system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

Replace `<your-username>` and `<your-repo-name>` with your actual GitHub
username and the repo name you created. It'll ask you to sign in the first
time (or use a personal access token if prompted).

## Notes

- The dataset here is **synthetic**, generated to have plausible statistical
  relationships between the health metrics and diabetes outcome — it isn't
  real patient data.
- This app is for educational purposes only and is not a medical diagnostic
  tool.
