import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model - UPDATED MODEL NAME
model = joblib.load('diabetes_model_v1_20250429.pkl')

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
# Convert to numerical values (must match training encoding!)
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
    # Prepare input data in EXACT SAME ORDER as model was trained
    input_data = np.array([
        age, bmi, waist_circ, bp_systolic, bp_diastolic,
        cholesterol_total, cholesterol_hdl, cholesterol_ldl,
        ggt, serum_urate, calories,
        encode_dict['Previous_Gestational_Diabetes'][gestational_diabetes],
        fasting_glucose, hba1c,
        encode_dict['Family_History_of_Diabetes'][family_history],
        encode_dict['Sex'][sex],
        encode_dict['Smoking_Status'][smoking_status],
        encode_dict['Alcohol_Consumption'][alcohol_consumption],
        encode_dict['Physical_Activity_Level'][physical_activity],
        encode_dict['Ethnicity'][ethnicity]
    ]).reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        # Display results
        if prediction == 1:
            st.error(f"🚨 High diabetes risk ({probability:.1f}% probability)")
            st.markdown("**Recommendations:** Consult a doctor for further evaluation.")
        else:
            st.success(f"✅ Low diabetes risk ({probability:.1f}% probability)")
            st.markdown("**Recommendations:** Maintain healthy lifestyle habits.")
        
        # Show raw probabilities (optional)
        with st.expander("Detailed Probabilities"):
            st.write(f"Probability of No Diabetes: {100 - probability:.1f}%")
            st.write(f"Probability of Diabetes: {probability:.1f}%")
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.info("Please ensure:")
        st.write("- All fields are filled correctly")
        st.write("- The model file exists and is compatible")
        st.write("- Input data matches the model's training format")
