import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model():
    """Load the trained model and associated files."""
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 'models', 'heart_disease_model.joblib')
        model_files = joblib.load(model_path)
        return model_files
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def validate_input(prompt, input_type, valid_range=None, valid_values=None):
    """Validate user input based on type and constraints."""
    while True:
        try:
            value = input(prompt)
            
            if input_type == "float":
                value = float(value)
                if valid_range:
                    if not (valid_range[0] <= value <= valid_range[1]):
                        print(f"Value must be between {valid_range[0]} and {valid_range[1]}")
                        continue
            elif input_type == "choice":
                value = value.upper()
                if valid_values and value not in valid_values:
                    print(f"Please enter one of: {', '.join(valid_values)}")
                    continue
            
            return value
        except ValueError:
            if input_type == "float":
                print("Please enter a valid number")
            else:
                print("Invalid input, please try again")

def get_user_input():
    """Get user input for prediction with validation."""
    print("\n=== Heart Disease Prediction System ===")
    print("Please enter the following information:\n")
    
    try:
        # Age validation (reasonable range: 18-100)
        age = validate_input(
            "Enter age (18-100): ",
            "float",
            valid_range=(18, 100)
        )

        # Sex validation
        sex = validate_input(
            "Enter sex (M/F): ",
            "choice",
            valid_values=["M", "F"]
        )

        # Chest pain type validation
        print("\nChest Pain Types:")
        print("TA  - Typical Angina")
        print("ATA - Atypical Angina")
        print("NAP - Non-Anginal Pain")
        print("ASY - Asymptomatic")
        chest_pain = validate_input(
            "Enter chest pain type (TA/ATA/NAP/ASY): ",
            "choice",
            valid_values=["TA", "ATA", "NAP", "ASY"]
        )

        # Resting BP validation (normal range: 90-200)
        resting_bp = validate_input(
            "Enter resting blood pressure (90-200 mm Hg): ",
            "float",
            valid_range=(90, 200)
        )

        # Cholesterol validation (normal range: 100-600)
        cholesterol = validate_input(
            "Enter cholesterol level (100-600 mg/dl): ",
            "float",
            valid_range=(100, 600)
        )

        # Fasting BS validation
        print("\nFasting Blood Sugar > 120 mg/dl?")
        fasting_bs = validate_input(
            "Enter Y for Yes, N for No: ",
            "choice",
            valid_values=["Y", "N"]
        )

        # Resting ECG validation
        print("\nResting ECG Types:")
        print("Normal - Normal")
        print("ST     - Having ST-T wave abnormality")
        print("LVH    - Showing probable or definite left ventricular hypertrophy")
        resting_ecg = validate_input(
            "Enter resting ECG (Normal/ST/LVH): ",
            "choice",
            valid_values=["NORMAL", "ST", "LVH"]
        )

        # Maximum heart rate validation (normal range: 60-220)
        max_hr = validate_input(
            "Enter maximum heart rate (60-220 bpm): ",
            "float",
            valid_range=(60, 220)
        )

        # Exercise-induced angina validation
        exercise_angina = validate_input(
            "Enter exercise-induced angina (Y/N): ",
            "choice",
            valid_values=["Y", "N"]
        )

        # ST depression validation (normal range: 0-6)
        oldpeak = validate_input(
            "Enter ST depression (0-6 mm): ",
            "float",
            valid_range=(0, 6)
        )

        # ST slope validation
        print("\nST Slope Types:")
        print("Up   - Upsloping")
        print("Flat - Flat")
        print("Down - Downsloping")
        st_slope = validate_input(
            "Enter ST slope (Up/Flat/Down): ",
            "choice",
            valid_values=["UP", "FLAT", "DOWN"]
        )

        # Convert categorical inputs
        sex = 1 if sex == 'M' else 0
        chest_pain_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        chest_pain = chest_pain_map[chest_pain]
        fasting_bs = 1 if fasting_bs == 'Y' else 0
        resting_ecg_map = {'NORMAL': 0, 'ST': 1, 'LVH': 2}
        resting_ecg = resting_ecg_map[resting_ecg]
        exercise_angina = 1 if exercise_angina == 'Y' else 0
        st_slope_map = {'UP': 0, 'FLAT': 1, 'DOWN': 2}
        st_slope = st_slope_map[st_slope]

        return pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })

    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

def preprocess_input(data, model_files):
    """Preprocess the input data."""
    try:
        # Create dummy variables
        data_encoded = pd.get_dummies(data)
        
        # Ensure all features from training are present
        for feature in model_files['feature_names']:
            if feature not in data_encoded.columns:
                data_encoded[feature] = 0

        # Reorder columns to match training data
        data_encoded = data_encoded[model_files['feature_names']]
        
        # Scale numerical features
        data_encoded[model_files['numerical_features']] = model_files['scaler'].transform(
            data_encoded[model_files['numerical_features']]
        )
        
        # Select only the features used in training
        selected_data = data_encoded[model_files['selected_features']]
        
        # Ensure column order matches training data
        selected_data = selected_data[model_files['X_train_columns']]
        
        return selected_data
    
    except Exception as e:
        print(f"\nError in preprocessing: {str(e)}")
        sys.exit(1)

def make_prediction(data, model_files):
    """Make prediction using the trained model."""
    try:
        prediction = model_files['model'].predict(data)
        probability = model_files['model'].predict_proba(data)
        return prediction[0], probability[0][1]
    except Exception as e:
        print(f"\nError making prediction: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the prediction system."""
    # Load model
    print("Loading model...")
    model_files = load_model()
    
    while True:
        try:
            # Get user input
            input_data = get_user_input()
            
            # Preprocess input
            processed_data = preprocess_input(input_data, model_files)
            
            # Make prediction
            prediction, probability = make_prediction(processed_data, model_files)
            
            # Display results
            print("\n=== Prediction Results ===")
            print(f"Heart Disease Risk: {'High' if prediction == 1 else 'Low'}")
            print(f"Probability: {probability:.2%}")
            
            # Ask if user wants to make another prediction
            again = validate_input(
                "\nWould you like to make another prediction? (Y/N): ",
                "choice",
                valid_values=["Y", "N"]
            )
            if again != 'Y':
                break

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            again = validate_input(
                "\nWould you like to try again? (Y/N): ",
                "choice",
                valid_values=["Y", "N"]
            )
            if again != 'Y':
                break

    print("\nThank you for using the Heart Disease Prediction System!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1) 