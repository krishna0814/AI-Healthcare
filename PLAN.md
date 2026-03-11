# AI-Based Disease Prediction System - Project Plan

## 📋 Project Overview
An AI-powered web application that predicts diseases (Diabetes, Heart, Liver) based on user-provided symptoms and medical reports using machine learning classification algorithms.

## 🛠 Tech Stack
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask (Python)
- **AI/ML**: Scikit-learn
- **Database**: MySQL
- **ML Algorithms**: Random Forest, Decision Tree, Logistic Regression

## 📁 Project Structure
```
AI-Healthcare-System/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── models/
│   │   ├── diabetes_model.pkl
│   │   ├── heart_model.pkl
│   │   └── liver_model.pkl
│   ├── datasets/
│   │   ├── diabetes.csv
│   │   ├── heart.csv
│   │   └── liver.csv
│   ├── train_models.py       # Model training script
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── index.html            # Home page
│   ├── diabetes.html         # Diabetes prediction page
│   ├── heart.html            # Heart prediction page
│   ├── liver.html            # Liver prediction page
│   ├── css/
│   │   └── style.css         # Main stylesheet
│   └── js/
│       └── main.js           # Frontend JavaScript
├── database/
│   └── schema.sql            # Database schema
└── README.md                 # Project documentation
```

## 🎯 Features to Implement

### 1. Disease Prediction
- **Diabetes Prediction**: Based on glucose, blood pressure, BMI, age, etc.
- **Heart Prediction**: Based on chest pain, cholesterol, ECG, etc.
- **Liver Prediction**: Based on bilirubin, enzyme levels, etc.

### 2. AI Features
- Symptom analysis with input validation
- Prediction confidence score (probability percentage)
- Multiple algorithm comparison (Random Forest, Decision Tree, Logistic Regression)
- Model accuracy display

### 3. User Interface
- Clean, medical-themed UI
- Input forms for each disease type
- Results display with confidence scores
- History of predictions (stored in MySQL)

### 4. Backend API
- RESTful API endpoints for each disease prediction
- POST /predict/diabetes
- POST /predict/heart
- POST /predict/liver
- GET /history - User prediction history

## 📊 ML Model Specifications

### Diabetes Dataset Features
- Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0=No Diabetes, 1=Diabetes)

### Heart Dataset Features
- Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- Target: HeartDisease (0=No, 1=Yes)

### Liver Dataset Features
- Age, Sex, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphatase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio
- Target: Target (1=Liver Disease, 2=No Liver Disease)

## 🔄 Implementation Steps

### Phase 1: Backend Setup
1. Create Flask application structure
2. Set up virtual environment and install dependencies
3. Create database schema
4. Create ML model training script
5. Train and save models

### Phase 2: API Development
1. Implement prediction endpoints
2. Add database integration
3. Add prediction history functionality

### Phase 3: Frontend Development
1. Create HTML pages for each disease
2. Style with CSS (medical theme)
3. Add JavaScript for API calls
4. Display results with confidence scores

### Phase 4: Testing & Integration
1. Test all prediction endpoints
2. Verify frontend-backend integration
3. Check database operations
4. Final UI polish

## 📝 Notes
- Use sample datasets for training (will be created with realistic medical data)
- Confidence scores will be calculated using predict_proba()
- Model accuracy will be displayed to users
- All predictions will be logged to MySQL database

