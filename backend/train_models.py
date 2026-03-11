"""
AI-Based Disease Prediction System - Model Training Script
Trains ML models for Diabetes, Heart Disease, and Liver Disease prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

# Create models directory if not exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_diabetes_data():
    """Load and prepare diabetes dataset"""
    print("\n" + "="*60)
    print("Loading Diabetes Dataset...")
    print("="*60)
    
    df = pd.read_csv(os.path.join(DATASETS_DIR, 'diabetes.csv'))
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: Outcome")
    print(f"Class distribution:\n{df['Outcome'].value_counts()}")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y, list(X.columns)

def load_heart_data():
    """Load and prepare heart disease dataset"""
    print("\n" + "="*60)
    print("Loading Heart Disease Dataset...")
    print("="*60)
    
    df = pd.read_csv(os.path.join(DATASETS_DIR, 'heart.csv'))
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: HeartDisease")
    print(f"Class distribution:\n{df['HeartDisease'].value_counts()}")
    
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    return X, y, list(X.columns)

def load_liver_data():
    """Load and prepare liver disease dataset"""
    print("\n" + "="*60)
    print("Loading Liver Disease Dataset...")
    print("="*60)
    
    df = pd.read_csv(os.path.join(DATASETS_DIR, 'liver.csv'))
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: Target")
    print(f"Class distribution:\n{df['Target'].value_counts()}")
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    return X, y, list(X.columns)

def train_and_evaluate_model(X, y, model_name, feature_names):
    """Train and evaluate multiple models for a disease"""
    print(f"\n{'='*60}")
    print(f"Training {model_name} Models...")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to try
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    results = {}
    
    for name, model in models.items():
        # Train model
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"\n{name}:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest Model: {best_model_name} with accuracy: {best_accuracy:.4f}")
    
    # Retrain best model on full training data
    if best_model_name == 'Logistic Regression':
        best_model.fit(X_train_scaled, y_train)
    else:
        best_model.fit(X_train, y_train)
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, f'{model_name.lower()}_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")
    
    return best_model, best_model_name, best_accuracy, results

def train_diabetes_model():
    """Train Diabetes prediction model"""
    X, y, feature_names = load_diabetes_data()
    model, model_name, accuracy, results = train_and_evaluate_model(
        X, y, 'diabetes', feature_names
    )
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    return model, accuracy, results

def train_heart_model():
    """Train Heart Disease prediction model"""
    X, y, feature_names = load_heart_data()
    model, model_name, accuracy, results = train_and_evaluate_model(
        X, y, 'heart', feature_names
    )
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'heart_model.pkl')
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    return model, accuracy, results

def train_liver_model():
    """Train Liver Disease prediction model"""
    X, y, feature_names = load_liver_data()
    model, model_name, accuracy, results = train_and_evaluate_model(
        X, y, 'liver', feature_names
    )
    
    # Save model
    model_path = os.path.join(MODELS_DIR, 'liver_model.pkl')
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")
    
    return model, accuracy, results

def print_summary(diabetes_acc, heart_acc, liver_acc):
    """Print training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Diabetes Model Accuracy:  {diabetes_acc:.4f}")
    print(f"Heart Model Accuracy:     {heart_acc:.4f}")
    print(f"Liver Model Accuracy:     {liver_acc:.4f}")
    print("="*60)
    print("\nAll models saved successfully!")
    print(f"Models directory: {MODELS_DIR}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("AI-Based Disease Prediction System")
    print("Model Training Script")
    print("="*60)
    
    # Train all models
    diabetes_model, diabetes_acc, diabetes_results = train_diabetes_model()
    heart_model, heart_acc, heart_results = train_heart_model()
    liver_model, liver_acc, liver_results = train_liver_model()
    
    # Print summary
    print_summary(diabetes_acc, heart_acc, liver_acc)
    
    print("\nTraining completed successfully!")

