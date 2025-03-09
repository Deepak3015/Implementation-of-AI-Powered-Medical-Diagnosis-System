🩺 Medical Diagnosis Using AI
Welcome to the Medical Diagnosis Using AI project! 🚀 This Streamlit application leverages machine learning models to predict the likelihood of various diseases based on user-provided data. The app features an intuitive interface with disease-specific input forms, predictions with confidence scores, and visually appealing images for each disease. 🌟

📋 Table of Contents
✨ Features

🛠️ Installation

🚀 Usage

🌐 Deployment

📂 Project Structure

🤝 Contributing

📜 License

🙏 Acknowledgments

✨ Features
Multi-Disease Prediction: 🩺 Supports diagnosis for six diseases with tailored input fields.

Machine Learning Models: 🤖 Utilizes pre-trained models (e.g., Random Forest) for accurate predictions.

Visual Interface: 🖼️ Displays disease-specific images and a welcoming home image.

Confidence Scores: 📊 Provides probability estimates for predictions.

User-Friendly: 🖱️ Simple sidebar selection and input forms with predefined ranges.

Educational Disclaimer: ⚠️ Includes a note to consult healthcare professionals.

🛠️ Installation
Prerequisites
Python 3.7 or higher 🐍

pip (Python package manager) 📦

Steps
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/medical-diagnosis-app.git
cd medical-diagnosis-app
(Replace yourusername/medical-diagnosis-app with your actual repository URL.)

Install Dependencies:

bash
Copy
pip install -r requirements.txt
If requirements.txt is not present, create it with:

bash
Copy
pip freeze > requirements.txt
Ensure it includes at least:

Copy
streamlit
pandas
scikit-learn
Verify Model Files: Ensure all model files are in the project directory:

Copy
Asthama_Severity_Mild_model.pkl
Asthama_Severity_Moderate_model.pkl
Asthama_Severity_None_model.pkl
breast_cancer_model.pkl
Chronic_Kidney_Dsease_.pkl
diabetes_model.pkl
Heart_diseases.pkl
Liver_diseases_data.pkl
Verify Image Files: Ensure all image files are in the project directory:

Copy
home_medical.jpg
asthma_lungs.jpeg
breast_cancer_ribbon.jpg
kidney_disease.jpg
diabetes_glucose.jpg
heart_disease.jpeg
liver_disease.jpg
(Images should be resized to 1920x640 pixels for optimal display.)

🚀 Usage
Run the App Locally:

bash
Copy
cd /home/ichigo/Desktop/Medical diagnosis uisng AI
streamlit run diseases.py
Access the App: Open your browser and go to http://localhost:8501. The home page displays a general medical image.

Use the Sidebar: Select a disease from the sidebar.

Input Data: Enter values for the required features (e.g., age, blood pressure) within the provided ranges. For Asthma, select binary options (0 = No, 1 = Yes).

Get Prediction: Click the "Predict" button to see the diagnosis (e.g., "HeartDisease Present" or "Predicted Severity: Mild") with a confidence score.

Explore Other Diseases: Change the disease selection to test different models and inputs.

🌐 Deployment
To deploy the app online using Streamlit Community Cloud:

Create a GitHub Repository: Initialize a new repository (e.g., medical-diagnosis-app) on GitHub.

Add all files (diseases.py, model files, image files, requirements.txt, and this README.md).

Push to GitHub:

bash
Copy
git init
git add .
git commit -m "Initial commit with app and models"
git remote add origin https://github.com/yourusername/medical-diagnosis-app.git
git push -u origin main
If model files exceed 100MB, use Git LFS:

bash
Copy
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add *.pkl
git commit -m "Add model files with LFS"
git push -u origin main
Deploy on Streamlit Community Cloud:

Go to streamlit.io/cloud.

Sign in with your GitHub account.

Click "New app" → "From existing repo."

Select your repository and set the main file path to diseases.py.

Click "Deploy" and wait for the app to go live.

Access the Deployed App: Use the provided URL (e.g., https://yourappname.streamlit.app).

📂 Project Structure
Copy
Medical diagnosis uisng AI/
├── diseases.py                  # Main Streamlit app code
├── Asthama_Severity_Mild_model.pkl      # Asthma Mild model
├── Asthama_Severity_Moderate_model.pkl  # Asthma Moderate model
├── Asthama_Severity_None_model.pkl      # Asthma None model
├── breast_cancer_model.pkl              # Breast Cancer model
├── Chronic_Kidney_Dsease_.pkl           # Chronic Kidney Disease model
├── diabetes_model.pkl                   # Diabetes model
├── Heart_diseases.pkl                   # Heart Disease model
├── Liver_diseases_data.pkl              # Liver Diseases model
├── home_medical.jpg                     # Home page image
├── asthma_lungs.jpeg                    # Asthma image
├── breast_cancer_ribbon.jpg             # Breast Cancer image
├── kidney_disease.jpg                   # Chronic Kidney Disease image
├── diabetes_glucose.jpg                 # Diabetes image
├── heart_disease.jpeg                   # Heart Disease image
├── liver_disease.jpg                    # Liver Diseases image
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
🤝 Contributing
Contributions are welcome! To contribute:

Fork the repository.

Create a new branch (git checkout -b feature-branch).

Make your changes and commit them (git commit -m "Add new feature").

Push to the branch (git push origin feature-branch).

Open a pull request.

Please ensure your code follows the project’s style and includes tests if applicable.

📜 License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it, but please include the original license.

🙏 Acknowledgments
Built with Streamlit, a fantastic framework for data apps. 🚀

Inspired by the need for accessible medical diagnosis tools. 🩺

Thanks to the open-source community for machine learning libraries like scikit-learn. 🤖

Note: Replace the placeholder image URL with a real screenshot of your app once you have one. Update the GitHub repository URL with your actual repo link. Add any additional acknowledgments or contributors as needed.
