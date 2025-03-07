import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Title of the app
st.title("Medical Diagnosis Using AI")

# Sidebar for disease selection
st.sidebar.header("Select Disease")
disease_options = [
    "Asthma", "Breast Cancer", "Chronic Kidney Disease", "Diabetes",
    "Heart Disease", "Liver Diseases"
]
selected_disease = st.sidebar.selectbox("Choose a disease", disease_options)

# Define model file paths
model_paths = {
    "Asthma": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/Asthama_model.pkl",
    "Breast Cancer": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/breast_cancer_model.pkl",
    "Chronic Kidney Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/Chronic_Kidney_Dsease_.pkl",
    "Diabetes": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/diabetes_model.pkl",
    "Heart Disease": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/Heart_diseases.pkl",
    "Liver Diseases": "/home/ichigo/Desktop/Medical diagnosis uisng AI/Models/Liver_diseases_data.pkl"
}

# Load the appropriate model using pickle
try:
    with open(model_paths[selected_disease], 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Model file {model_paths[selected_disease]} not found. Please check the path and filename.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define feature columns for each disease (excluding target)
feature_columns = {
    "Asthma": ['Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing', 'Sore-Throat',
               'Pains', 'Nasal-Congestion', 'Runny-Nose', 'Age_0-9', 'Age_10-19',
               'Age_20-24', 'Age_25-59', 'Age_60+', 'Gender_Female', 'Gender_Male',
               'Severity_Mild', 'Severity_Moderate', 'Severity_None'],
    "Breast Cancer": ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                      'smoothness_mean', 'compactness_mean', 'concavity_mean',
                      'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                      'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                      'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                      'fractal_dimension_se', 'radius_worst', 'texture_worst',
                      'perimeter_worst', 'area_worst', 'smoothness_worst',
                      'compactness_worst', 'concavity_worst', 'concave points_worst',
                      'symmetry_worst', 'fractal_dimension_worst'],
    "Chronic Kidney Disease": ['Age', 'Gender', 'Ethnicity', 'SocioeconomicStatus',
                              'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
                              'PhysicalActivity', 'DietQuality', 'SleepQuality',
                              'FamilyHistoryKidneyDisease', 'FamilyHistoryHypertension',
                              'FamilyHistoryDiabetes', 'PreviousAcuteKidneyInjury',
                              'UrinaryTractInfections', 'SystolicBP', 'DiastolicBP',
                              'FastingBloodSugar', 'HbA1c', 'SerumCreatinine', 'BUNLevels',
                              'GFR', 'ProteinInUrine', 'ACR', 'SerumElectrolytesSodium',
                              'SerumElectrolytesPotassium', 'SerumElectrolytesCalcium',
                              'SerumElectrolytesPhosphorus', 'HemoglobinLevels',
                              'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                              'CholesterolTriglycerides', 'ACEInhibitors', 'Diuretics',
                              'NSAIDsUse', 'Statins', 'AntidiabeticMedications', 'Edema',
                              'FatigueLevels', 'NauseaVomiting', 'MuscleCramps', 'Itching',
                              'QualityOfLifeScore', 'HeavyMetalsExposure',
                              'OccupationalExposureChemicals', 'WaterQuality',
                              'MedicalCheckupsFrequency', 'MedicationAdherence',
                              'HealthLiteracy'],
    "Diabetes": ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                 'BMI', 'DiabetesPedigreeFunction', 'Age'],
    "Heart Disease": ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang',
                      'oldpeak', 'ca', 'thal'],
    "Liver Diseases": ['Age', 'Gender', 'BMI', 'AlcoholConsumption', 'Smoking',
                       'GeneticRisk', 'PhysicalActivity', 'Diabetes', 'Hypertension',
                       'LiverFunctionTest']
}

# Create input form based on selected disease
st.header(f"Enter {selected_disease} Data")
input_data = {}

for column in feature_columns[selected_disease]:
    if 'Age' in column or any(sub in column.lower() for sub in ['bp', 'pressure', 'sugar', 'creatinine', 'bun', 'gfr', 'ac', 'electrolytes', 'hemoglobin', 'cholesterol', 'triglycerides']):
        input_data[column] = st.number_input(f"{column} (numeric)", value=0.0, step=0.1)
    elif 'Gender' in column or any(sub in column.lower() for sub in ['male', 'female', 'smoking', 'diabetes', 'hypertension', 'risk']):
        input_data[column] = st.selectbox(f"{column}", [0, 1])
    elif any(sub in column.lower() for sub in ['severity', 'ethnicity', 'education', 'family', 'previous', 'urinary', 'ace', 'diuretics', 'nsaids', 'statins', 'antidiabetic', 'edema', 'fatigue', 'nausea', 'cramps', 'itching', 'quality', 'exposure', 'water', 'medical', 'medication', 'health']):
        input_data[column] = st.selectbox(f"{column}", [0, 1, 2] if 'risk' in column.lower() else [0, 1])
    else:
        input_data[column] = st.number_input(f"{column} (numeric)", value=0.0, step=0.1)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical features (assuming model expects scaled data)
scaler = StandardScaler()
numerical_cols = [col for col in feature_columns[selected_disease] if any(sub in col.lower() for sub in ['age', 'bmi', 'alcohol', 'physical', 'liver', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'symmetry', 'fractal', 'trestbps', 'chol', 'thalach', 'oldpeak'])]
input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]
        diagnosis = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Diagnosis: {diagnosis}")
        st.write(f"Confidence: {probability[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Display feature importance with length check
if hasattr(model, 'feature_importances_'):
    num_features_model = len(model.feature_importances_)
    num_features_defined = len(feature_columns[selected_disease])
    st.write(f"Number of features in model: {num_features_model}")
    st.write(f"Number of features defined: {num_features_defined}")
    if num_features_model == num_features_defined:
        importance = pd.DataFrame({
            'Feature': feature_columns[selected_disease],
            'Importance': model.feature_importances_
        })
        st.subheader("Feature Importance")
        st.bar_chart(importance.set_index('Feature'))
    else:
        st.warning(f"Feature count mismatch: Model expects {num_features_model} features, but {num_features_defined} are defined. Adjust feature_columns.")