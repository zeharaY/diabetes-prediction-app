import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Improved model loading function
def load_model(model_path):
    try:
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'model' in model_data:
            return model_data['model']  # Extract the actual model from dictionary
        return model_data  # Return directly if it's already a model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

# Load the trained model
model = load_model('diabetes_model.pkl')

# Verify model has predict method
if not hasattr(model, 'predict'):
    st.error("Loaded object is not a valid prediction model")
    st.stop()

# App title
st.title("Diabetes Risk Prediction App 🩺")
st.markdown("Enter patient details to assess diabetes risk:")

# --- Categorical Inputs (with encoding) ---
st.header("Demographic Information")
col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Sex", ["Female", "Male"])
    ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "White"])
with col2:
    smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["None", "Moderate", "Heavy"])

physical_activity = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

# --- Numerical Inputs ---
st.header("Medical Measurements")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 18, 120, 30)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    waist_circ = st.number_input("Waist Circumference (cm)", 50, 200, 80)
    bp_systolic = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
    bp_diastolic = st.number_input("Diastolic BP (mmHg)", 40, 120, 80)
with col2:
    cholesterol_total = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
    cholesterol_hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 50)
    cholesterol_ldl = st.number_input("LDL Cholesterol (mg/dL)", 50, 300, 100)
    ggt = st.number_input("GGT (U/L)", 5, 200, 25)
    serum_urate = st.number_input("Serum Urate (mg/dL)", 2.0, 10.0, 5.0)

st.header("Lifestyle & History")
calories = st.number_input("Daily Caloric Intake", 800, 5000, 2000)
gestational_diabetes = st.selectbox("Previous Gestational Diabetes", ["No", "Yes"])
fasting_glucose = st.number_input("Fasting Blood Glucose (mg/dL)", 50, 300, 90)
hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# --- Encoding Categorical Variables ---
encode_dict = {
    'Sex': {"Female": 0, "Male": 1},
    'Smoking_Status': {"Never": 0, "Former": 1, "Current": 2},
    'Alcohol_Consumption': {"None": 0, "Moderate": 1, "Heavy": 2},
    'Physical_Activity_Level': {"Low": 0, "Moderate": 1, "High": 2},
    'Ethnicity': {"Asian": 0, "Black": 1, "Hispanic": 2, "White": 3},
    'Previous_Gestational_Diabetes': {"No": 0, "Yes": 1},
    'Family_History_of_Diabetes': {"No": 0, "Yes": 1}
}

# --- Prediction Button ---
if st.button("Calculate Diabetes Risk"):
    try:
        # Create DataFrame with named columns
        input_data = pd.DataFrame({
            'Age': [age],
            'BMI': [bmi],
            'Waist_Circumference': [waist_circ],
            'Blood_Pressure_Systolic': [bp_systolic],
            'Blood_Pressure_Diastolic': [bp_diastolic],
            'Cholesterol_Total': [cholesterol_total],
            'Cholesterol_HDL': [cholesterol_hdl],
            'Cholesterol_LDL': [cholesterol_ldl],
            'GGT': [ggt],
            'Serum_Urate': [serum_urate],
            'Dietary_Intake_Calories': [calories],
            'Previous_Gestational_Diabetes': [encode_dict['Previous_Gestational_Diabetes'][gestational_diabetes]],
            'Fasting_Blood_Glucose': [fasting_glucose],
            'HbA1c': [hba1c],
            'Family_History_of_Diabetes': [encode_dict['Family_History_of_Diabetes'][family_history]],
            'Sex': [encode_dict['Sex'][sex]],
            'Smoking_Status': [encode_dict['Smoking_Status'][smoking_status]],
            'Alcohol_Consumption': [encode_dict['Alcohol_Consumption'][alcohol_consumption]],
            'Physical_Activity_Level': [encode_dict['Physical_Activity_Level'][physical_activity]],
            'Ethnicity': [encode_dict['Ethnicity'][ethnicity]]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Display results
        if prediction == 1:
            st.error(f"🚨 High diabetes risk ({probability:.1f}% probability)")
            st.markdown("**Recommendations:** Consult a doctor for further evaluation.")
        else:
            st.success(f"✅ Low diabetes risk ({probability:.1f}% probability)")
            st.markdown("**Recommendations:** Maintain healthy lifestyle habits.")

        with st.expander("Detailed Probabilities"):
            st.write(f"Probability of No Diabetes: {100 - probability:.1f}%")
            st.write(f"Probability of Diabetes: {probability:.1f}%")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Troubleshooting tips:")
        st.write("- Verify all fields are complete")
        st.write("- Check model expects same features in this order")
        st.write("- Confirm model file wasn't corrupted")
