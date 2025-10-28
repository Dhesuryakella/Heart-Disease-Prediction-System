import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os

def train_models(data_path=None):
    """Train and evaluate multiple models on the heart disease dataset."""
    
    if data_path is None:
        # Get the absolute path to the data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_path = os.path.join(project_root, 'data', 'processed', 'heart.csv')
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']
    
    # Preprocess data
    X_encoded = pd.get_dummies(X)
    feature_names = X_encoded.columns.tolist()  # Store as list instead of Index
    
    # Scale numerical features
    numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    scaler = StandardScaler()
    X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
    
    # Feature selection
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X_encoded, y)
    selected_features = X_encoded.columns[selector.get_support()].tolist()  # Store as list
    
    # Create a DataFrame with selected features to preserve feature names
    X_selected = pd.DataFrame(X_selected, columns=selected_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier()
    }
    
    # Train and evaluate models
    results = {}
    best_score = 0
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        score = model.score(X_test, y_test)
        print(f"Test Score: {score:.4f}")
        
        # Save results
        results[name] = {
            'model': model,
            'cv_score': cv_scores.mean(),
            'test_score': score
        }
        
        # Track best model
        if score > best_score:
            best_score = score
            best_model = name
    
    # Save best model
    print(f"\nBest model: {best_model} (Test Score: {best_score:.4f})")
    
    # Create model files dictionary
    model_files = {
        'model': results[best_model]['model'],
        'scaler': scaler,
        'feature_names': feature_names,
        'numerical_features': numerical_features,
        'selected_features': selected_features,
        'X_train_columns': X_train.columns.tolist()  # Save training columns order
    }
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model files
    model_path = os.path.join(models_dir, 'heart_disease_model.joblib')
    joblib.dump(model_files, model_path)
    print(f"\nModel saved as '{model_path}'")
    
    return results, model_files

if __name__ == "__main__":
    results, model_files = train_models() 