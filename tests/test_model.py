import unittest
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_model import train_models

class TestHeartDiseaseModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and train model if needed."""
        cls.data_path = os.path.join('data', 'processed', 'heart.csv')
        cls.model_path = os.path.join('models', 'heart_disease_model.joblib')
        
        # Train model if it doesn't exist
        if not os.path.exists(cls.model_path):
            cls.results, cls.model_files = train_models(cls.data_path)
        else:
            cls.model_files = joblib.load(cls.model_path)
    
    def test_model_exists(self):
        """Test if the model file exists."""
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_model_prediction(self):
        """Test model prediction on a sample case."""
        # Sample case
        sample_case = pd.DataFrame([{
            'Age': 63,
            'Sex': 'M',
            'ChestPainType': 'ASY',
            'RestingBP': 145,
            'Cholesterol': 233,
            'FastingBS': 1,
            'RestingECG': 'Normal',
            'MaxHR': 150,
            'ExerciseAngina': 'N',
            'Oldpeak': 2.3,
            'ST_Slope': 'Down'
        }])
        
        # Preprocess sample case
        sample_encoded = pd.get_dummies(sample_case)
        
        # Ensure all features from training are present
        for feature in self.model_files['feature_names']:
            if feature not in sample_encoded.columns:
                sample_encoded[feature] = 0
        
        # Reorder columns to match training data
        sample_encoded = sample_encoded[self.model_files['feature_names']]
        
        # Scale numerical features
        sample_encoded[self.model_files['numerical_features']] = \
            self.model_files['scaler'].transform(sample_encoded[self.model_files['numerical_features']])
        
        # Make prediction
        prediction = self.model_files['model'].predict(sample_encoded)
        
        # Check prediction type and range
        self.assertIsInstance(prediction[0], (np.int64, int))
        self.assertIn(prediction[0], [0, 1])
    
    def test_feature_importance(self):
        """Test if feature importance is available and valid."""
        model = self.model_files['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            self.assertEqual(len(importances), len(self.model_files['selected_features']))
            self.assertTrue(all(isinstance(x, float) for x in importances))
            self.assertAlmostEqual(sum(importances), 1.0, places=5)

if __name__ == '__main__':
    unittest.main() 