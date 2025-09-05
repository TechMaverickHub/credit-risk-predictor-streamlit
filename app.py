import streamlit as st
import pandas as pd
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    preprocess = joblib.load("models/preprocess.joblib")
    model = joblib.load("models/xgb_model.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    metadata = joblib.load("models/metadata.joblib")
    return preprocess, model, feature_names, metadata

preprocess, model, feature_names, metadata = load_artifacts()

# Prediction function
def predict_risk(input_dict):
    df = pd.DataFrame([input_dict])

    # 1. For Employment Length
    df['emp_length_missing'] = df['person_emp_length'].isna().astype(int)


    # Transform
    X_proc = preprocess.transform(df)

    # Re-align to saved feature names
    X_proc = pd.DataFrame(X_proc, columns=feature_names)

    # Predict
    pred = model.predict(X_proc)[0]
    prob = model.predict_proba(X_proc)[0, 1]
    return pred, prob

# ----------------------
# Streamlit UI
# ----------------------
st.title("💳 Credit Risk Prediction App")

st.sidebar.header("📌 Enter Applicant Details")

# Numeric inputs
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=1_000_000, value=50_000)
person_emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=metadata['medians'].get('person_emp_length'))
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=500, max_value=100_000, value=10_000)
loan_int_rate = st.sidebar.slider("Loan Interest Rate (%)", 1.0, 30.0, metadata['medians'].get('loan_int_rate'))
loan_percent_income = st.sidebar.slider("Loan Percent of Income (%)", 0.01, 1.0, 0.2)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (years)", min_value=0, max_value=50, value=10)

# Binary inputs
cb_person_default_on_file = st.sidebar.selectbox(
    "Default on File", 
    options=[0, 1], 
    format_func=lambda x: "Yes" if x==1 else "No"
)

# Categorical (OneHot)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_intent = st.sidebar.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

# Ordinal
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

# Assemble input dict
input_dict = {
    "person_age": person_age,
    "person_income": person_income,
    "person_emp_length": person_emp_length,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "cb_person_default_on_file": cb_person_default_on_file,
    "person_home_ownership": person_home_ownership,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade
}

# Predict button
if st.sidebar.button("Predict Credit Risk"):
    pred, prob = predict_risk(input_dict)
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ High Risk of Default (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {prob:.2f})")
