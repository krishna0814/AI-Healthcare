"""
AI-Based Disease Prediction System - Flask Backend
Predicts Diabetes, Heart Disease, and Liver Disease using ML models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load models
def load_models():
    """Load all trained models"""
    models = {}
    try:
        models['diabetes'] = joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
        models['heart'] = joblib.load(os.path.join(MODELS_DIR, 'heart_model.pkl'))
        models['liver'] = joblib.load(os.path.join(MODELS_DIR, 'liver_model.pkl'))
        
        # Load scalers if they exist
        models['diabetes_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl'))
        models['heart_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'heart_scaler.pkl'))
        models['liver_scaler'] = joblib.load(os.path.join(MODELS_DIR, 'liver_scaler.pkl'))
    except FileNotFoundError:
        print("Warning: Models not found. Please run train_models.py first.")
    return models

# Global models dictionary
MODELS = load_models()

# Database simulation (for demo purposes - uses in-memory storage)
# In production, replace with MySQL connection
prediction_history = []

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI-Based Disease Prediction System API',
        'version': '1.0.0',
        'endpoints': {
            'predict_diabetes': '/predict/diabetes',
            'predict_heart': '/predict/heart',
            'predict_liver': '/predict/liver',
            'history': '/history',
            'models_status': '/models/status'
        }
    })

@app.route('/models/status')
def models_status():
    """Check if models are loaded"""
    return jsonify({
        'diabetes': 'diabetes_model.pkl' in os.listdir(MODELS_DIR),
        'heart': 'heart_model.pkl' in os.listdir(MODELS_DIR),
        'liver': 'liver_model.pkl' in os.listdir(MODELS_DIR)
    })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """
    Predict Diabetes based on input features
    Features: Pregnancies, Glucose, BloodPressure, SkinThickness, 
              Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodPressure', 0)),
            float(data.get('skinThickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('diabetesPedigreeFunction', 0)),
            float(data.get('age', 0))
        ]
        
        # Create DataFrame with feature names
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        X = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        if 'diabetes' in MODELS:
            model = MODELS['diabetes']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Calculate confidence score
            confidence = float(max(probability)) * 100
            
            result = {
                'prediction': int(prediction),
                'result': 'Positive (Diabetes Detected)' if prediction == 1 else 'Negative (No Diabetes)',
                'confidence': round(confidence, 2),
                'probability': {
                    'negative': round(float(probability[0]) * 100, 2),
                    'positive': round(float(probability[1]) * 100, 2)
                }
            }
        else:
            # Fallback if model not loaded
            result = {
                'prediction': -1,
                'result': 'Model not loaded',
                'confidence': 0,
                'message': 'Please run train_models.py first'
            }
        
        # Save to history
        history_entry = {
            'disease': 'Diabetes',
            'input': data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.append(history_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    """
    Predict Heart Disease based on input features
    Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
              RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
    """
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('age', 0)),
            int(data.get('sex', 0)),
            int(data.get('chestPainType', 0)),
            float(data.get('restingBP', 0)),
            float(data.get('cholesterol', 0)),
            int(data.get('fastingBS', 0)),
            int(data.get('restingECG', 0)),
            float(data.get('maxHR', 0)),
            int(data.get('exerciseAngina', 0)),
            float(data.get('oldpeak', 0)),
            int(data.get('ST_Slope', 0))
        ]
        
        # Create DataFrame with feature names
        feature_names = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                        'Oldpeak', 'ST_Slope']
        X = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        if 'heart' in MODELS:
            model = MODELS['heart']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Calculate confidence score
            confidence = float(max(probability)) * 100
            
            result = {
                'prediction': int(prediction),
                'result': 'Positive (Heart Disease Detected)' if prediction == 1 else 'Negative (No Heart Disease)',
                'confidence': round(confidence, 2),
                'probability': {
                    'negative': round(float(probability[0]) * 100, 2),
                    'positive': round(float(probability[1]) * 100, 2)
                }
            }
        else:
            result = {
                'prediction': -1,
                'result': 'Model not loaded',
                'confidence': 0,
                'message': 'Please run train_models.py first'
            }
        
        # Save to history
        history_entry = {
            'disease': 'Heart Disease',
            'input': data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.append(history_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/liver', methods=['POST'])
def predict_liver():
    """
    Predict Liver Disease based on input features
    Features: Age, Sex, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphatase,
              Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
              Albumin, Albumin_and_Globulin_Ratio
    """
    try:
        data = request.get_json()
        
        # Extract features
        features = [
            float(data.get('age', 0)),
            int(data.get('sex', 0)),
            float(data.get('totalBilirubin', 0)),
            float(data.get('directBilirubin', 0)),
            float(data.get('alkalinePhosphatase', 0)),
            float(data.get('alamineAminotransferase', 0)),
            float(data.get('aspartateAminotransferase', 0)),
            float(data.get('totalProtiens', 0)),
            float(data.get('albumin', 0)),
            float(data.get('albuminAndGlobulinRatio', 0))
        ]
        
        # Create DataFrame with feature names
        feature_names = ['Age', 'Sex', 'Total_Bilirubin', 'Direct_Bilirubin', 
                        'Alkaline_Phosphatase', 'Alamine_Aminotransferase', 
                        'Aspartate_Aminotransferase', 'Total_Protiens', 
                        'Albumin', 'Albumin_and_Globulin_Ratio']
        X = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        if 'liver' in MODELS:
            model = MODELS['liver']
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            # Calculate confidence score
            confidence = float(max(probability)) * 100
            
            result = {
                'prediction': int(prediction),
                'result': 'Positive (Liver Disease Detected)' if prediction == 1 else 'Negative (No Liver Disease)',
                'confidence': round(confidence, 2),
                'probability': {
                    'no_disease': round(float(probability[0]) * 100, 2),
                    'liver_disease': round(float(probability[1]) * 100, 2)
                }
            }
        else:
            result = {
                'prediction': -1,
                'result': 'Model not loaded',
                'confidence': 0,
                'message': 'Please run train_models.py first'
            }
        
        # Save to history
        history_entry = {
            'disease': 'Liver Disease',
            'input': data,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        prediction_history.append(history_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    return jsonify({
        'history': prediction_history,
        'total': len(prediction_history)
    })

@app.route('/history', methods=['DELETE'])
def clear_history():
    """Clear prediction history"""
    prediction_history.clear()
    return jsonify({'message': 'History cleared successfully'})

if __name__ == '__main__':
    print("Starting AI-Based Disease Prediction System...")
    print("Loading models...")
    MODELS = load_models()
    print(f"Models loaded: {list(MODELS.keys())}")
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)

