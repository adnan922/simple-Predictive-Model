import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# Load saved models
log_model = pickle.load(open('model/log_model.pkl', 'rb'))
dt_model = pickle.load(open('model/dt_model.pkl', 'rb'))
rf_model = pickle.load(open('model/rf_model.pkl', 'rb'))


# Title
st.title("Iris Species Prediction")

# Sidebar for Model Selection
st.sidebar.header("Choose Model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])

# Sidebar for Input Features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

# Create feature array
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction logic
if model_choice == "Logistic Regression":
    model = log_model
elif model_choice == "Decision Tree":
    model = dt_model
else:
    model = rf_model

# Make prediction and calculate probabilities
pred = model.predict(input_features)
prob = model.predict_proba(input_features)

# Map prediction to species
species = ['Setosa', 'Versicolor', 'Virginica']

# Display prediction result
st.subheader("Prediction Result:")
st.write(f"Predicted Species: **{species[pred[0]]}**")

# Display prediction probabilities
st.subheader("Prediction Probabilities:")
prob_df = pd.DataFrame(prob, columns=species)
st.write(prob_df)

# Model Information
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.text(f"Model: {model_choice}")
st.sidebar.text(f"Classes: {species}")

# Footer
st.markdown("---")
st.text("Developed with Streamlit")
