
import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np

# Load and preprocess the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(df[data.feature_names])
y = df['target']

# Train a simple ANN model
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=500, random_state=42)
mlp.fit(X, y)

# Streamlit App
st.title("Breast Cancer Prediction App")
st.write(''' 
### Explore the Breast Cancer Dataset and Predict Cancer Type
''')

# Dataset exploration
if st.sidebar.checkbox("Show Dataset"):
    st.write(df.head())

# User Input Section
st.sidebar.write("### Enter Features for Prediction")
user_input = []
for feature in data.feature_names:
    value = st.sidebar.number_input(feature, min_value=float(np.min(df[feature])), max_value=float(np.max(df[feature])), value=float(df[feature].mean()))
    user_input.append(value)

# Prediction Button
if st.sidebar.button("Predict"):
    # Scale user input and make a prediction
    user_input_scaled = scaler.transform([user_input])
    prediction = mlp.predict(user_input_scaled)
    prediction_proba = mlp.predict_proba(user_input_scaled)
    
    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.write(f"### Prediction: {result}")
    st.write(f"Probability of Malignant: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of Benign: {prediction_proba[0][0]:.2f}")

# Visualization Section
st.sidebar.write("### Dataset Visualization")
if st.sidebar.checkbox("Show Target Distribution"):
    st.write("Target Distribution in Dataset")
    st.bar_chart(df['target'].value_counts())
