import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Breast Cancer ANN", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Detection ANN")
st.markdown("Artificial Neural Network (MLP) trained on the Wisconsin Breast Cancer Dataset.")

# ============================================================
# CACHED DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names

x, y, feature_names, target_names = load_data()

# ============================================================
# SIDEBAR - HYPERPARAMETERS
# ============================================================
with st.sidebar:
    st.header("⚙️ ANN Hyperparameters")
    st.write("Tune the Multi-Layer Perceptron (MLP) architecture and parameters.")
    
    hidden_layer_1 = st.slider("Hidden Layer 1 Neurons", 8, 256, 128, step=8)
    hidden_layer_2 = st.slider("Hidden Layer 2 Neurons", 8, 256, 64, step=8)
    learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.01, 0.1], index=1)
    max_iter = st.slider("Max Iterations", 100, 1000, 500, step=100)
    
    train_button = st.button("🚀 Retrain Model", type="primary", use_container_width=True)

# ============================================================
# MAIN AREA
# ============================================================
st.subheader("📊 Dataset Overview")
st.write(f"**Dataset Shape:** `{x.shape[0]} samples`, `{x.shape[1]} features`")
st.write(f"**Classes:** `0 = {target_names[0].capitalize()} (Malignant)`, `1 = {target_names[1].capitalize()} (Benign)`")

# Data preparation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (reacts to sidebar inputs)
with st.spinner("Training Neural Network..."):
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
        activation='relu',
        solver='adam', 
        learning_rate_init=learning_rate,
        max_iter=max_iter,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)

# Predictions
y_pred = mlp.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.divider()
st.subheader("📈 Performance Metrics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy*100:.2f}%")
m2.metric("Precision", f"{precision*100:.2f}%")
m3.metric("Recall", f"{recall*100:.2f}%")
m4.metric("F1-Score", f"{f1*100:.2f}%")

st.divider()
st.subheader("🩺 Test a Random Patient Sample")
st.write("Since this dataset requires 30 different medical measurements, you can pick a random patient from the test set to see how the model evaluates them!")

if st.button("🎲 Pick Random Patient"):
    # Pick a random sample from the test set
    idx = np.random.randint(0, len(X_test))
    patient_data = X_test[idx]
    true_label = y_test[idx]
    
    # Predict
    pred_label = mlp.predict(scaler.transform([patient_data]))[0]
    
    true_str = target_names[true_label].capitalize()
    pred_str = target_names[pred_label].capitalize()
    
    col_a, col_b = st.columns(2)
    col_a.info(f"**True Diagnosis:** `{true_str}`")
    col_b.info(f"**ANN Prediction:** `{pred_str}`")
    
    if true_label == pred_label:
        st.success("✅ The Neural Network correctly diagnosed this patient!")
    else:
        st.error("❌ The Neural Network made an incorrect diagnosis.")
