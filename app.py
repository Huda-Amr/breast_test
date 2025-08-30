import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@st.cache_resource
def train_and_save_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, data.feature_names

model, feature_names = train_and_save_model()

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the required values. The model is trained on the sklearn breast cancer dataset.")

selected_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

inputs = []
for feature in selected_features:
    val = st.number_input(f"Enter {feature}", min_value=0.0, value=10.0)
    inputs.append(val)

if st.button("🔍 Predict"):
    X_input = np.zeros((1, len(feature_names)))
    for i, feature in enumerate(selected_features):
        col_index = list(feature_names).index(feature)
        X_input[0, col_index] = inputs[i]

    prediction = model.predict(X_input)[0]
    if prediction == 0:
        st.success("✅ Prediction: Benign")
    else:
        st.error("⚠️ Prediction: Malignant")
