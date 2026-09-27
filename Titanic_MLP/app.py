import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Titanic MLP Predictor", 
    page_icon="🚢", 
    layout="centered"
)

# ============================================================
# ML PIPELINE (CACHED)
# ============================================================
@st.cache_resource
def train_model():
    # 1. Load Dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # 2. Preprocess Data
    # Fill missing age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Fill missing embarked with most common
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Map categorical data
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Scale Data (MLPs are very sensitive to feature scaling!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        max_iter=1000, 
        random_state=42,
        early_stopping=True
    )
    mlp.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    y_pred = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return mlp, scaler, acc

# Load model in background
with st.spinner("Training Multi-Layer Perceptron on Titanic dataset..."):
    model, scaler, accuracy = train_model()

# ============================================================
# UI 
# ============================================================
st.title("🚢 Titanic Fate Predictor")
st.markdown("Enter your passenger details below to see if you would survive the Titanic disaster, powered by an AI neural network (MLP)!")
st.caption(f"🤖 Model Accuracy on Test Data: {accuracy * 100:.1f}%")

st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass_str = st.selectbox("Ticket Class", ["1st Class", "2nd Class", "3rd Class"])
    pclass = int(pclass_str[0])  # Extract 1, 2, or 3
    
    sex_str = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if sex_str == "Female" else 0
    
    age = st.slider("Age", 0, 100, 30)
    
    embarked_str = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
    embarked = embarked_map[embarked_str]

with col2:
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
    fare = st.slider("Ticket Fare ($)", 0.0, 500.0, 32.0)

st.write("")
predict_btn = st.button("🔮 Predict My Fate", type="primary", use_container_width=True)

if predict_btn:
    # Prepare input vector
    input_data = pd.DataFrame([[
        pclass, sex, age, sibsp, parch, fare, embarked
    ]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    
    # Scale and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.divider()
    
    if prediction == 1:
        st.success(f"🎉 **YOU SURVIVED!**")
        st.markdown(f"The neural network predicts a **{probability[1]*100:.1f}%** chance of survival.")
        st.balloons()
    else:
        st.error(f"☠️ **YOU DID NOT SURVIVE.**")
        st.markdown(f"The neural network predicts a **{probability[0]*100:.1f}%** chance of perishing.")
