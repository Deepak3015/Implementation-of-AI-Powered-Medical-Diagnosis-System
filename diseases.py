import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# App title
st.title("Medical Diagnosis Using AI")

# Sidebar for disease selection
disease = st.sidebar.selectbox("Select Disease", ["Asthma", "Breast Cancer", "Chronic Kidney Disease", "Diabetes", "Heart Disease", "Liver Diseases"])

# Model paths
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
    "Liver Diseases": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Liver_diseases_data.pkl"
}

# Load models
models = {s: pickle.load(open(p, 'rb')) for s, p in model_paths["Asthma"].items()} if disease == "Asthma" else {disease: pickle.load(open(model_paths[disease], 'rb'))}

# Feature columns
features = {
    "Asthma": ['Difficulty-in-Breathing', 'Dry-Cough', 'Pains', 'Tiredness', 'Age_0-9'],
    "Breast Cancer": ['concave points_worst', 'perimeter_worst', 'area_worst', 'radius_worst', 'concave points_mean', 'perimeter_mean', 'area_mean', 'radius_mean', 'concavity_mean', 'concavity_worst'],
    "Chronic Kidney Disease": ['GFR', 'SerumCreatinine', 'Age', 'SystolicBP', 'BMI', 'ProteinInUrine', 'DiastolicBP', 'HemoglobinLevels', 'FastingBloodSugar', 'FamilyHistoryKidneyDisease'],
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "Heart Disease": ['ca', 'thal', 'cp', 'oldpeak', 'thalach', 'exang', 'age'],
    "Liver Diseases": ['LiverFunctionTest', 'BMI', 'AlcoholConsumption', 'Diabetes', 'Age']
}

# Input ranges for numerical features
input_ranges = {
    "Asthma": {col: (0, 1) for col in features["Asthma"]},  # Binary inputs
    "Breast Cancer": {  # Typical ranges for breast cancer features (based on UCI Breast Cancer dataset)
        'concave points_worst': (0.0, 0.3), 'perimeter_worst': (50.0, 250.0), 'area_worst': (200.0, 2500.0),
        'radius_worst': (7.0, 36.0), 'concave points_mean': (0.0, 0.2), 'perimeter_mean': (40.0, 200.0),
        'area_mean': (150.0, 2000.0), 'radius_mean': (6.0, 30.0), 'concavity_mean': (0.0, 0.5), 'concavity_worst': (0.0, 1.3)
    },
    "Chronic Kidney Disease": {  # Based on typical medical ranges
        'GFR': (0, 150), 'SerumCreatinine': (0.0, 10.0), 'Age': (0, 120), 'SystolicBP': (70, 200),
        'BMI': (15.0, 50.0), 'ProteinInUrine': (0.0, 5.0), 'DiastolicBP': (40, 120),
        'HemoglobinLevels': (5.0, 20.0), 'FastingBloodSugar': (50, 300), 'FamilyHistoryKidneyDisease': (0, 1)
    },
    "Diabetes": {  # Based on UCI Diabetes dataset
        'Pregnancies': (0, 20), 'Glucose': (0, 200), 'BloodPressure': (0, 150), 'SkinThickness': (0, 100),
        'Insulin': (0, 900), 'BMI': (0.0, 70.0), 'DiabetesPedigreeFunction': (0.0, 2.5), 'Age': (0, 120)
    },
    "Heart Disease": {'ca': (0, 4), 'thal': (1, 3), 'cp': (0, 3), 'oldpeak': (0.0, 6.0), 'thalach': (60, 220), 'exang': (0, 1), 'age': (20, 100)},
    "Liver Diseases": {  # Reasonable ranges for liver disease features
        'LiverFunctionTest': (0.0, 100.0), 'BMI': (15.0, 50.0), 'AlcoholConsumption': (0, 100), 'Diabetes': (0, 1), 'Age': (0, 120)
    }
}

# Categorical options
categorical_options = {
    "Asthma": {col: {0: "No", 1: "Yes"} for col in features["Asthma"]},
    "Chronic Kidney Disease": {'FamilyHistoryKidneyDisease': {0: "No", 1: "Yes"}},
    "Heart Disease": {'cp': {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}, 'exang': {0: "No", 1: "Yes"}, 'thal': {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}},
    "Liver Diseases": {'Diabetes': {0: "No", 1: "Yes"}}
}

# Input form
st.header(f"Enter {disease} Data")
input_data = {}
for col in features[disease]:
    if col in categorical_options.get(disease, {}):
        options = categorical_options[disease][col]
        input_data[col] = st.selectbox(col, list(options.keys()), format_func=lambda x: options[x])
    else:
        min_val, max_val = input_ranges[disease].get(col, (0.0, 100.0))
        input_data[col] = st.number_input(col, min_value=float(min_val), max_value=float(max_val), value=float((min_val + max_val) / 2))

input_df = pd.DataFrame([input_data])
input_df_scaled = StandardScaler().fit_transform(input_df)

# Prediction
if st.button("Predict"):
    probs = {s: m.predict_proba(input_df)[:, 1][0] for s, m in models.items()} if disease == "Asthma" else {disease: models[disease].predict_proba(input_df_scaled)[:, 1][0]}
    pred = max(probs, key=probs.get) if disease == "Asthma" else models[disease].predict(input_df_scaled)[0]
    result = f"Predicted Severity: {pred.capitalize()} (Confidence: {probs[pred]:.2f})" if disease == "Asthma" else f"Diagnosis: {disease.replace(' ', '')} {'Present' if pred == 1 else 'Absent'} (Confidence: {probs[disease]:.2f})"
    st.success(result)
    if disease == "Asthma": st.dataframe(pd.DataFrame({"Severity": probs.keys(), "Probability": [f"{p:.2f}" for p in probs.values()]}))

# Disclaimer
st.write("**Disclaimer**: For educational purposes only. Consult a healthcare professional.")