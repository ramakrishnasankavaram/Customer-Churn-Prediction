import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Load the encoders and scaler
try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders/scaler: {e}")
    st.stop()

# App title and description
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title('📊 Customer Churn Prediction App')
st.write("Fill in the customer details below to predict the likelihood of churn.")

# Input fields
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.slider('📅 Age', 18, 92, value=30)
balance = st.number_input('💰 Balance', value=0.0, step=100.0)
credit_score = st.number_input('📈 Credit Score', value=600, step=10)
estimated_salary = st.number_input('💸 Estimated Salary', value=50000.0, step=1000.0)
tenure = st.slider('📆 Tenure', 0, 10, value=5)
num_of_products = st.slider('📦 Number of Products', 1, 4, value=2)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x else "No")
is_active_member = st.selectbox('🔄 Is Active Member?', [0, 1], format_func=lambda x: "Yes" if x else "No")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
try:
    input_data_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()

# Predict churn
try:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
except Exception as e:
    st.error(f"Error making predictions: {e}")
    st.stop()

# Display results
st.write("### Results")
st.write("---")
st.metric("Churn Probability", f"{prediction_proba:.2%}")

if prediction_proba > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')
