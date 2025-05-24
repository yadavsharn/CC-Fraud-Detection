# webapp/app.py
import streamlit as st
import pandas as pd
import joblib
from utils import preprocess

# Load model
model = joblib.load('model/fraud_model.pkl')

st.title("💳 Credit Card Fraud Detection")

option = st.radio("Choose input method:", ["Manual Input", "Upload CSV"])

if option == "Manual Input":
    st.subheader("Enter Transaction Details")
    V_features = [st.number_input(f"V{i}", step=0.1) for i in range(1, 29)]
    amount = st.number_input("Amount", step=0.01)

    if st.button("Predict"):
        input_df = pd.DataFrame([V_features + [amount]], columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])
        input_df = preprocess(input_df)  # Preprocess the input similarly to training data
        prediction = model.predict(input_df)[0]
        st.success("✅ Legitimate Transaction" if prediction == 0 else "⚠️ Fraudulent Transaction")

else:
    st.subheader("Upload CSV File")
    file = st.file_uploader("Upload your input CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = preprocess(df)
        # Remove the target column if present
        if 'Class' in df.columns:
            X = df.drop(columns=['Class'])
        else:
            X = df
        predictions = model.predict(X)
        df['Prediction'] = predictions
        st.write(df)
        st.success("✅ All transactions processed.")
