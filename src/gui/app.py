import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class HeartDiseasePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Predictor")
        self.root.geometry("800x900")
        
        # Configure style
        style = ttk.Style()
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(self.main_frame, text="Heart Disease Prediction System", 
                              font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Create input fields
        self.create_input_fields()
        
        # Create buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.grid(row=12, column=0, columnspan=2, pady=20)
        
        # Create predict button
        predict_btn = ttk.Button(buttons_frame, text="Predict", command=self.predict)
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Create result display area
        self.create_result_area()
        
    def create_input_fields(self):
        # Patient Information
        ttk.Label(self.main_frame, text="Patient Information", 
                 font=("Helvetica", 12, "bold")).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Age
        ttk.Label(self.main_frame, text="Age:").grid(row=2, column=0, sticky=tk.W)
        self.age_var = tk.StringVar(value="50")
        age_entry = ttk.Entry(self.main_frame, textvariable=self.age_var)
        age_entry.grid(row=2, column=1, sticky=tk.W)
        
        # Sex
        ttk.Label(self.main_frame, text="Sex:").grid(row=3, column=0, sticky=tk.W)
        self.sex_var = tk.StringVar(value="M")
        sex_combo = ttk.Combobox(self.main_frame, textvariable=self.sex_var, values=["M", "F"])
        sex_combo.grid(row=3, column=1, sticky=tk.W)
        
        # Chest Pain Type
        ttk.Label(self.main_frame, text="Chest Pain Type:").grid(row=4, column=0, sticky=tk.W)
        self.cp_var = tk.StringVar(value="ASY")
        cp_combo = ttk.Combobox(self.main_frame, textvariable=self.cp_var, 
                               values=["ASY", "ATA", "NAP", "TA"])
        cp_combo.grid(row=4, column=1, sticky=tk.W)
        
        # Resting BP
        ttk.Label(self.main_frame, text="Resting BP (mmHg):").grid(row=5, column=0, sticky=tk.W)
        self.bp_var = tk.StringVar(value="120")
        bp_entry = ttk.Entry(self.main_frame, textvariable=self.bp_var)
        bp_entry.grid(row=5, column=1, sticky=tk.W)
        
        # Cholesterol
        ttk.Label(self.main_frame, text="Cholesterol (mg/dl):").grid(row=6, column=0, sticky=tk.W)
        self.chol_var = tk.StringVar(value="200")
        chol_entry = ttk.Entry(self.main_frame, textvariable=self.chol_var)
        chol_entry.grid(row=6, column=1, sticky=tk.W)
        
        # Fasting BS
        ttk.Label(self.main_frame, text="Fasting Blood Sugar > 120 mg/dl:").grid(row=7, column=0, sticky=tk.W)
        self.fbs_var = tk.StringVar(value="0")
        fbs_combo = ttk.Combobox(self.main_frame, textvariable=self.fbs_var, values=["0", "1"])
        fbs_combo.grid(row=7, column=1, sticky=tk.W)
        
        # Resting ECG
        ttk.Label(self.main_frame, text="Resting ECG:").grid(row=8, column=0, sticky=tk.W)
        self.ecg_var = tk.StringVar(value="Normal")
        ecg_combo = ttk.Combobox(self.main_frame, textvariable=self.ecg_var, 
                                values=["Normal", "ST", "LVH"])
        ecg_combo.grid(row=8, column=1, sticky=tk.W)
        
        # Max HR
        ttk.Label(self.main_frame, text="Maximum Heart Rate:").grid(row=9, column=0, sticky=tk.W)
        self.hr_var = tk.StringVar(value="150")
        hr_entry = ttk.Entry(self.main_frame, textvariable=self.hr_var)
        hr_entry.grid(row=9, column=1, sticky=tk.W)
        
        # Exercise Angina
        ttk.Label(self.main_frame, text="Exercise Angina:").grid(row=10, column=0, sticky=tk.W)
        self.angina_var = tk.StringVar(value="N")
        angina_combo = ttk.Combobox(self.main_frame, textvariable=self.angina_var, 
                                   values=["Y", "N"])
        angina_combo.grid(row=10, column=1, sticky=tk.W)
        
        # Oldpeak
        ttk.Label(self.main_frame, text="ST Depression (Oldpeak):").grid(row=11, column=0, sticky=tk.W)
        self.oldpeak_var = tk.StringVar(value="0.0")
        oldpeak_entry = ttk.Entry(self.main_frame, textvariable=self.oldpeak_var)
        oldpeak_entry.grid(row=11, column=1, sticky=tk.W)
        
        # ST Slope
        ttk.Label(self.main_frame, text="ST Slope:").grid(row=12, column=0, sticky=tk.W)
        self.slope_var = tk.StringVar(value="Up")
        slope_combo = ttk.Combobox(self.main_frame, textvariable=self.slope_var, 
                                  values=["Up", "Flat", "Down"])
        slope_combo.grid(row=12, column=1, sticky=tk.W)
        
    def create_result_area(self):
        # Results Frame
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Prediction Results", padding="10")
        self.result_frame.grid(row=13, column=0, columnspan=2, pady=20, sticky=(tk.W, tk.E))
        
        # Result Labels
        self.diagnosis_var = tk.StringVar(value="---")
        self.risk_var = tk.StringVar(value="---")
        self.prob_var = tk.StringVar(value="---")
        
        ttk.Label(self.result_frame, text="Diagnosis:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(self.result_frame, textvariable=self.diagnosis_var).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(self.result_frame, text="Risk Level:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(self.result_frame, textvariable=self.risk_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(self.result_frame, text="Probability:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(self.result_frame, textvariable=self.prob_var).grid(row=2, column=1, sticky=tk.W)
        
        # Risk Factors Text Area
        self.risk_factors_text = tk.Text(self.result_frame, height=6, width=40)
        self.risk_factors_text.grid(row=3, column=0, columnspan=2, pady=10)
        
    def predict(self):
        try:
            # Gather input values
            patient_data = {
                'Age': int(self.age_var.get()),
                'Sex': self.sex_var.get(),
                'ChestPainType': self.cp_var.get(),
                'RestingBP': int(self.bp_var.get()),
                'Cholesterol': int(self.chol_var.get()),
                'FastingBS': int(self.fbs_var.get()),
                'RestingECG': self.ecg_var.get(),
                'MaxHR': int(self.hr_var.get()),
                'ExerciseAngina': self.angina_var.get(),
                'Oldpeak': float(self.oldpeak_var.get()),
                'ST_Slope': self.slope_var.get()
            }
            
            # Load model and make prediction
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                    'models', 'heart_disease_model.joblib')
            model_files = joblib.load(model_path)
            result = self.predict_heart_disease(patient_data, model_files)
            
            # Update result display
            self.diagnosis_var.set(result['prediction'])
            self.risk_var.set(result['risk_level'])
            self.prob_var.set(f"{result['probability']:.1%}")
            
            # Update risk factors
            self.risk_factors_text.delete(1.0, tk.END)
            risk_factors = self.identify_risk_factors(patient_data)
            if risk_factors:
                self.risk_factors_text.insert(tk.END, "Risk Factors Present:\n\n")
                for factor in risk_factors:
                    self.risk_factors_text.insert(tk.END, f"• {factor}\n")
            else:
                self.risk_factors_text.insert(tk.END, "No major risk factors identified")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def predict_heart_disease(self, sample_case, model_files):
        model = model_files['model']
        scaler = model_files['scaler']
        feature_names = model_files['feature_names']
        numerical_features = model_files['numerical_features']
        selected_features = model_files['selected_features']
        
        # Create a DataFrame with the sample case
        sample_df = pd.DataFrame([sample_case])
        
        # Convert categorical variables to match training data format
        sample_df['Sex'] = sample_df['Sex'].map({'M': 1, 'F': 0})
        sample_df['ChestPainType'] = sample_df['ChestPainType'].map({'TA': 'TA', 'ATA': 'ATA', 'NAP': 'NAP', 'ASY': 'ASY'})
        sample_df['RestingECG'] = sample_df['RestingECG'].map({'Normal': 'Normal', 'ST': 'ST', 'LVH': 'LVH'})
        sample_df['ExerciseAngina'] = sample_df['ExerciseAngina'].map({'Y': 'Y', 'N': 'N'})
        sample_df['ST_Slope'] = sample_df['ST_Slope'].map({'Up': 'Up', 'Flat': 'Flat', 'Down': 'Down'})
        
        # One-hot encode categorical variables
        sample_df_encoded = pd.get_dummies(sample_df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
        
        # Ensure all features from training are present
        for feature in feature_names:
            if feature not in sample_df_encoded.columns:
                sample_df_encoded[feature] = 0
        
        # Reorder columns to match training data
        sample_df_encoded = sample_df_encoded[feature_names]
        
        # Scale numerical features
        sample_df_encoded[numerical_features] = scaler.transform(sample_df_encoded[numerical_features])
        
        # Select only the features used in training
        sample_df_selected = sample_df_encoded[selected_features]
        
        # Make prediction
        prediction = model.predict(sample_df_selected)[0]
        probability = model.predict_proba(sample_df_selected)[0][1]
        
        return {
            'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Moderate' if probability > 0.3 else 'Low'
        }
    
    def identify_risk_factors(self, data):
        risk_factors = []
        if int(data['Age']) > 55:
            risk_factors.append("Age > 55")
        if data['Sex'] == 'M':
            risk_factors.append("Male gender")
        if data['ChestPainType'] == 'ASY':
            risk_factors.append("Asymptomatic chest pain")
        if int(data['RestingBP']) > 140:
            risk_factors.append("High blood pressure")
        if int(data['Cholesterol']) > 240:
            risk_factors.append("High cholesterol")
        if data['ExerciseAngina'] == 'Y':
            risk_factors.append("Exercise-induced angina")
        if float(data['Oldpeak']) > 2:
            risk_factors.append("High ST depression")
        if data['ST_Slope'] == 'Down':
            risk_factors.append("Downsloping ST segment")
        if int(data['MaxHR']) < 120:
            risk_factors.append("Low maximum heart rate")
        if data['RestingECG'] != 'Normal':
            risk_factors.append("Abnormal resting ECG")
        return risk_factors

def main():
    root = tk.Tk()
    app = HeartDiseasePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 