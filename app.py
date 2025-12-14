import collections.abc
# Fix for Python 3.10+ compatibility
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping


import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# 1. Load the "Translator" (Encoder) and "Brain" (Model)
encoder = joblib.load('encoder.joblib')
model = XGBClassifier()
model.load_model('model.json')

# 2. setup Page
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉")
st.title("📉 Customer Churn Prediction App")

# 3. Input Form
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

st.subheader("Billing Details")
col3, col4 = st.columns(2)
with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    # Add hidden defaults for columns you might have trained on but aren't asking user for
    device = "No" 
    tech_support = "No"
    streaming_tv = "No"
    streaming_movies = "No"
    
with col4:
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

# 4. PREPARE DATA (The Logic)
# 4. PREPARE DATA (The Logic)
if st.button("Predict Churn Risk"):
    
    # A. Create Raw DataFrame
    cat_data = pd.DataFrame({
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [security],
        "OnlineBackup": [backup],
        "DeviceProtection": [device],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment]
    })

    # B. Create Numerical DataFrame
    num_data = pd.DataFrame({
        "SeniorCitizen": [senior],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    # --- THE FIX STARTS HERE ---
    # 1. Ask the encoder: "What order do you want?" and automatically reorder columns
    # This prevents the "ValueError" about feature names/order
    try:
        cat_data = cat_data[encoder.feature_names_in_]
    except KeyError as e:
        st.error(f"Column name mismatch! The model expects: {encoder.feature_names_in_}")
        st.stop()
    # ---------------------------

    # C. Encode Categorical Data
    cat_encoded = pd.DataFrame(encoder.transform(cat_data))
    
    # --- FIX: MATCH THE "UGLY" NOTEBOOK NAMES ---
    # Your model was trained on columns named "0", "1", "2", etc.
    # So we must name them "0", "1", "2" here too.
    cat_encoded.columns = cat_encoded.columns.astype(str)
    # --------------------------------------------

    # D. Concatenate
    # This matches the order: [Numbers, 0, 1, 2...]
    final_input = pd.concat([num_data, cat_encoded], axis=1)
    
    # E. Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    st.write("---")
    if prediction == 1:
        st.error(f"🚨 **High Risk!** Probability: {probability:.1%}")
        # Optional: Show a "Why?" chart if you want, but get this working first!
    else:
        st.success(f"✅ **Safe.** Probability: {(1-probability):.1%}")