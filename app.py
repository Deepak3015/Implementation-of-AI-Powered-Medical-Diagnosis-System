import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# App title
st.title("Medical Diagnosis Using AI")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Select Disease", ["", "Asthma", "Breast Cancer", "Chronic Kidney Disease", "Diabetes", "Heart Disease", "Liver Diseases"])

# Image paths
images = {
    "": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/home_medical.jpg",
    "Asthma": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/asthma_lungs.jpeg",
    "Breast Cancer": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/breast_cancer_ribbon.jpg",
    "Chronic Kidney Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/kidney_disease.jpg",
    "Diabetes": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/diabetes_glucose.jpg",
    "Heart Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/heart_disease.jpeg",
    "Liver Diseases": "/home/ichigo/Desktop/Medical diagnosis uisng AI/images/liver_disease.jpg"
}

# Display image
try:
    st.image(images[disease] if disease else images[""], caption=f"{disease or 'Medical Diagnosis Home'} Image", width=None)
except Exception as e:
    st.error(f"Failed to load image: {str(e)}")

# Home screen prompt
if not disease:
    st.write("**Please select a disease from the sidebar to proceed.**")

# Model and scaler paths
model_paths = {
    "Asthma": {
        "Mild": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Asthama_Severity_Mild_model.pkl",
        "Moderate": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Asthama_Severity_Moderate_model.pkl",
        "None": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Asthama_Severity_None_model.pkl"
    },
    "Breast Cancer": "/home/ichigo/Desktop/Medical diagnosis uisng AI/breast_cancer_model.pkl",
    "Chronic Kidney Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Chronic_Kidney_Dsease_.pkl",
    "Diabetes": "/home/ichigo/Desktop/Medical diagnosis uisng AI/diabetes_model.pkl",
    "Heart Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Heart_diseases.pkl",
    "Liver Diseases": "/home/ichigo/Desktop/Medical diagnosis uisng AI/liver_model.pkl"
}

# Load models and scalers
models = {s: pickle.load(open(p, 'rb')) for s, p in model_paths["Asthma"].items()} if disease == "Asthma" else {disease: pickle.load(open(model_paths[disease], 'rb')) if disease else None}
diabetes_scaler = pickle.load(open('/home/ichigo/Desktop/Medical diagnosis uisng AI/diabetes_scaler.pkl', 'rb')) if disease == "Diabetes" else None
liver_scaler = pickle.load(open('/home/ichigo/Desktop/Medical diagnosis uisng AI/liver_scaler.pkl', 'rb')) if disease == "Liver Diseases" else None

# Feature columns
features = {
    "Asthma": ['Difficulty-in-Breathing', 'Dry-Cough', 'Pains', 'Tiredness', 'Age_0-9'],
    "Breast Cancer": ['concave points_worst', 'perimeter_worst', 'area_worst', 'radius_worst', 'concave points_mean', 'perimeter_mean', 'area_mean', 'radius_mean', 'concavity_mean', 'concavity_worst'],
    "Chronic Kidney Disease": ['GFR', 'SerumCreatinine', 'Age', 'SystolicBP', 'BMI', 'ProteinInUrine', 'DiastolicBP', 'HemoglobinLevels', 'FastingBloodSugar', 'FamilyHistoryKidneyDisease'],
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "Heart Disease": ['ca', 'thal', 'cp', 'oldpeak', 'thalach', 'exang', 'age'],
    "Liver Diseases": ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension', 'LiverFunctionTest']
}

# Input ranges (updated for Liver Diseases based on dataset)
input_ranges = {
    "Asthma": {col: (0, 1) for col in features["Asthma"]},
    "Breast Cancer": {
        'concave points_worst': (0.0, 0.3), 'perimeter_worst': (50.0, 250.0), 'area_worst': (200.0, 2500.0),
        'radius_worst': (7.0, 36.0), 'concave points_mean': (0.0, 0.2), 'perimeter_mean': (40.0, 200.0),
        'area_mean': (150.0, 2000.0), 'radius_mean': (6.0, 30.0), 'concavity_mean': (0.0, 0.5), 'concavity_worst': (0.0, 1.3)
    },
    "Chronic Kidney Disease": {
        'GFR': (0, 150), 'SerumCreatinine': (0.0, 10.0), 'Age': (0, 120), 'SystolicBP': (70, 200),
        'BMI': (15.0, 50.0), 'ProteinInUrine': (0.0, 5.0), 'DiastolicBP': (40, 120),
        'HemoglobinLevels': (5.0, 20.0), 'FastingBloodSugar': (50, 300), 'FamilyHistoryKidneyDisease': (0, 1)
    },
    "Diabetes": {
        'Pregnancies': (0, 20), 'Glucose': (0, 200), 'BloodPressure': (0, 150), 'SkinThickness': (0, 100),
        'Insulin': (0, 900), 'BMI': (0.0, 70.0), 'DiabetesPedigreeFunction': (0.0, 2.5), 'Age': (0, 120)
    },
    "Heart Disease": {'ca': (0, 4), 'thal': (1, 3), 'cp': (0, 3), 'oldpeak': (0.0, 6.0), 'thalach': (60, 220), 'exang': (0, 1), 'age': (20, 100)},
    "Liver Diseases": {
        'Age': (20, 80), 'Gender': (0, 1), 'BMI': (15.0, 40.0), 'AlcoholConsumption': (0.0, 20.0),
        'Smoking': (0, 1), 'GeneticRisk': (0, 2), 'PhysicalActivity': (0.0, 10.0),
        'Diabetes': (0, 1), 'Hypertension': (0, 1), 'LiverFunctionTest': (20.0, 100.0)
    }
}

# Input form
if disease:
    st.header(f"Enter {disease} Data")
    input_data = {}
    for col in features[disease]:
        min_val, max_val = input_ranges[disease].get(col, (0.0, 100.0))
        input_data[col] = st.number_input(col, min_value=float(min_val), max_value=float(max_val), value=float(min_val))

    input_df = pd.DataFrame([input_data])

    # Scale input for Diabetes or Liver Diseases
    if disease == "Diabetes" and diabetes_scaler:
        input_df_scaled = diabetes_scaler.transform(input_df)
    elif disease == "Liver Diseases" and liver_scaler:
        input_df_scaled = liver_scaler.transform(input_df)
    else:
        input_df_scaled = StandardScaler().fit_transform(input_df)

    # Prediction
    if st.button("Predict"):
        if disease == "Asthma":
            probs = {s: m.predict_proba(input_df)[:, 1][0] for s, m in models.items()}
            pred = max(probs, key=probs.get)
            result = f"Predicted Severity: {pred.capitalize()} (Confidence: {probs[pred]:.2f})"
            st.success(result)
            st.dataframe(pd.DataFrame({"Severity": probs.keys(), "Probability": [f"{p:.2f}" for p in probs.values()]}))
        else:
            probs = models[disease].predict_proba(input_df_scaled)[:, 1][0]
            pred = models[disease].predict(input_df_scaled)[0]
            result = f"Diagnosis: {disease.replace(' ', '')} {'Present' if pred == 1 else 'Absent'} (Confidence: {probs:.2f})"
            st.success(result)

# Disclaimer
st.write("**Disclaimer**: For educational purposes only. Consult a healthcare professional.")