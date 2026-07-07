# AI-Powered-Credit-Scoring-Income-Verification-for-Digital-Lending

A machine learning–powered REST API that predicts loan default risk for borrowers using an XGBoost classification model served through FastAPI.

Overview

This project takes borrower information (age, income, credit score, loan details, etc.), engineers a set of derived risk features, and returns a default-risk probability along with a human-readable recommendation (approve / review / decline).

Tech Stack


FastAPI — REST API framework
XGBoost — gradient-boosted tree model for risk prediction
scikit-learn — feature scaling (StandardScaler)
pandas / numpy — data handling
Pydantic — request/response validation
joblib — model serialization

How It Works

1. Input

The API accepts these borrower fields:

FieldDescriptionAgeBorrower's ageIncomeAnnual incomeLoanAmountRequested loan amountCreditScoreCredit scoreMonthsEmployedEmployment duration (months)NumCreditLinesNumber of open credit linesInterestRateLoan interest rate (%)LoanTermLoan term (months)DTIRatioDebt-to-income ratio

2. Feature Engineering

Before prediction, model_service.py derives additional features from the raw inputs, including:


Ratios: loan-to-income, monthly payment estimate, payment-to-income ratio
Interaction terms: age × income, interest rate × loan term
Risk flags: high-rate + low-credit, young + high-debt
Transforms: log(income), age², credit score³
Buckets: age group, income quartile, loan amount quartile


This mirrors the exact feature set the model was trained on.

3. Prediction

The scaled feature vector is passed to the trained XGBoost model, which returns a risk probability. This is converted into a risk category (e.g. Low / Medium / High) and a lending recommendation.

API Endpoints

MethodEndpointDescriptionGET/API infoGET/healthHealth check — confirms model is loadedPOST/predictPredict risk for a single borrowerPOST/predict/batchPredict risk for multiple borrowers at onceGET/model/performanceModel metrics and feature importanceGET/docsInteractive Swagger UI (auto-generated)

Setup & Run

bash# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API
uvicorn app.main:app --reload

# 3. Open the interactive docs
http://localhost:8000/docs

Example Request

jsonPOST /predict
{
  "Age": 35,
  "Income": 65000,
  "LoanAmount": 20000,
  "CreditScore": 700,
  "MonthsEmployed": 48,
  "NumCreditLines": 4,
  "InterestRate": 10.5,
  "LoanTerm": 60,
  "DTIRatio": 0.28
}

Example Response

json{
  "risk_score": 0.18,
  "risk_category": "LOW",
  "recommendation": "APPROVE",
  "factors": {
    "top_positive_factors": ["Good credit score", "Stable employment history"],
    "top_negative_factors": [],
    "risk_score_explanation": "Risk score of 18.0% based on 2 positive and 0 negative factors"
  },
  "model_version": "1.0.0"
}

Model Training

The model was trained covering:

Data loading and exploratory analysis
Preprocessing and categorical encoding
Feature engineering (see above)
Training and comparing Logistic Regression, Random Forest, XGBoost, and LightGBM
Model evaluation (classification report, confusion matrix, AUC-ROC, AUC-PR)
Ensemble stacking

Notes


This is a demo/portfolio project, not production-hardened — no authentication is implemented on the API endpoints.
