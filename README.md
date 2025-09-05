# Credit Risk Predictor

A simple Streamlit app to predict whether a loan applicant is likely to default.  
The model uses scikit-learn preprocessing and an XGBoost classifier.

---
## User Inputs in the App

The app collects applicant information through a simple sidebar form:

### Numeric Inputs
- **Person Age** (years)  
- **Person Income** (annual income in USD)  
- **Employment Length** (years in current job)  
- **Loan Amount** (amount requested)  
- **Loan Interest Rate (%)**  
- **Loan Percent Income** (ratio of loan payment to income)  
- **Credit History Length** (years)  

### Categorical Inputs
- **Home Ownership** (choices: RENT, OWN, MORTGAGE, OTHER)  
- **Loan Intent** (choices: EDUCATION, MEDICAL, VENTURE, PERSONAL, HOMEIMPROVEMENT, DEBTCONSOLIDATION)  
- **Loan Grade** (choices: A, B, C, D, E, F, G)  

### Binary Input
- **Previous Default on File** (Yes = 1, No = 0)  

All inputs are passed through preprocessing (imputation, encoding, and scaling if needed) before being fed into the model.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/TechMaverickHub/credit-risk-predictor-streamlit.git
cd credit-risk-predictor-streamlit

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.\.venv\Scripts\activate    # Windows

pip install -r requirements.txt

streamlit run app.py
