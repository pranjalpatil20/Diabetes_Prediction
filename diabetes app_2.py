import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# PAGE CONFIG-----
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🧠",
    layout="wide"
)

# LOAD MODEL-----
model=pickle.load(open("best_model1.pkl", "rb"))

# TITLE-----
st.title("🧠 Diabetes Risk Predictor")
st.markdown("### Real-time health risk analysis using Machine Learning")

# SIDEBAR INPUTS-----
st.sidebar.header("📝 Patient Details")

glucose = st.sidebar.slider("Blood Glucose Level", 70, 200, 100)
hba1c = st.sidebar.slider("HbA1c Level", 3.3, 15.0, 13.0)
age = st.sidebar.slider("Age", 10, 90, 30)
bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.0)

# Categorical inputs-----
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
smoking = st.sidebar.selectbox("Smoking History", ["never", "former", "current"])

# Binary inputs-----
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])

# CREATE INPUT DATAFRAME-----
input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking],
    "bmi": [bmi],
    "HbA1c_level": [hba1c],
    "blood_glucose_level": [glucose]
})

# IMPORTANT: INTERACTION FEATURE-----
input_data["glucose_hba1c_interaction"] = (
    input_data["blood_glucose_level"] * input_data["HbA1c_level"]
)

# PREDICTION-----
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

# DISPLAY RESULTS-----
st.subheader("📊 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

with col2:
    st.metric("🧠 Probability", f"{round(probability*100, 2)}%")

# PROGRESS BAR-----
st.progress(int(probability * 100))

# RISK LEVEL-----
if probability > 0.60:
    st.error("🔴 Critical Risk")
elif probability > 0.30:
    st.warning("🟠 Moderate Risk")
else:
    st.success("🟢 Safe Zone")    

# if 9 <= hba1c <=13:
    # st.success("🟢 Safe Zone")  
# else:
    # st.error("🔴 Critical Risk")
    
st.info("This result based on a machine learning model and is not a medical diagnosis. Please consult a healthcare professional for proper evaluation")    

# FOOTER-----
st.markdown("---")
st.caption("🤖Powered by Machine Learning | Built with Streamlit")
