import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle  # Import joblib to load the pre-trained model

def main():
    st.markdown("<h1 style='text-align: center; color: #15F4F4; text-decoration:underline;'>Bankruptcy Prevention</h1>", unsafe_allow_html=True)
    image = Image.open('bank.jpg')
    st.sidebar.image(image, width=300)

if __name__ == "__main__":
    main()

# Load the pre-trained model
with open("model_SVM", "rb") as file:
    model = pickle.load(file)

training_feature_names = ['management_risk', 'financial_flexibility', 'credibility', 'competitiveness']

# Function to make predictions
def predict_bankruptcy(training_feature_names):
    # Ensure the features have the same order as when you trained the model
    prediction = model.predict(training_feature_names)
    return prediction

# Your input form
st.sidebar.title("Predict Bankruptcy")
management_risk = st.number_input("Management Risk", min_value=0.0, max_value=1.0, step=0.01)
financial_flexibility = st.number_input("Financial Flexibility", min_value=0.0, max_value=1.0, step=0.01)
credibility = st.number_input("Credibility", min_value=0.0, max_value=1.0, step=0.01)
competitiveness = st.number_input("Competitiveness", min_value=0.0, max_value=1.0, step=0.01)

if st.button("Predict"):
    # Create a DataFrame from user input
    user_input = pd.DataFrame({
        ' management_risk': [management_risk],
        ' financial_flexibility': [financial_flexibility],
        ' credibility': [credibility],
        ' competitiveness': [competitiveness]
    })


    # Make a prediction
    prediction = predict_bankruptcy(user_input)

    # Display the result
    if prediction == 0:
        st.subheader("Business is likely to go bankrupt.")
        image = Image.open('bankrupt.png')
        st.image(image, width=300)
    else:
        st.subheader("Business is likely to survive.")
        image = Image.open('nonbankrupt1.jpg')
        st.image(image, width=300)