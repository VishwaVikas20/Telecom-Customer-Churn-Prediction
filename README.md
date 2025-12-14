# Telecom Customer Churn Prediction 📉

A Machine Learning project that predicts whether a telecom customer is likely to leave (churn) based on their subscription details.

## 👨‍💻 Project Overview
**My Role:** Data Cleaning, Feature Engineering, Model Training, & Model Serialization.
**Objective:** To build a robust model that processes raw data and saves a trained model for production use.

*Note: The `app.py` (Streamlit interface) in this repository was generated using AI tools to demonstrate the underlying model's functionality in a web browser.*

## 🧠 The "Brain" (My Work)
The core logic resides in `ML.ipynb`.
1. **Data Preprocessing:** Implemented label encoding for categorical variables (Gender, Contract Type, etc.) using `joblib` to save the encoder for consistent future use.
2. **Model Selection:** Utilized **XGBoost Classifier** for its efficiency and high performance on structured tabular data.
3. **Evaluation:** Validated model performance using Mean Absolute Error (MAE) and probability metrics.
4. **Serialization:** Successfully saved the trained model (`model.json`) and encoder (`encoder.joblib`) to simulate a real-world deployment pipeline.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, XGBoost, Joblib
* **Interface:** Streamlit

## 🚀 How to Run locally
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost streamlit joblib
