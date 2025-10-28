# Heart Disease Prediction System

A comprehensive system for predicting heart disease risk using machine learning.

## Features

- Interactive GUI for data input and predictions
- Multiple machine learning models (Logistic Regression, Random Forest, SVM, etc.)
- Automatic model selection based on performance
- Risk factor identification
- Real-time predictions
- Model training capability

## Setup

1. Install Python 3.7 or higher
2. Install required packages:

   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place your training data file (heart.csv) in the `data/processed/` directory
   - The CSV should have the following columns:
     - Age
     - Sex (M/F)
     - ChestPainType (ASY/ATA/NAP/TA)
     - RestingBP
     - Cholesterol
     - FastingBS
     - RestingECG (Normal/ST/LVH)
     - MaxHR
     - ExerciseAngina (Y/N)
     - Oldpeak
     - ST_Slope (Up/Flat/Down)
     - HeartDisease (0/1)

## Usage

1. Run the application:

   ```
   python heart_disease_system.py
   ```

2. The GUI will open, allowing you to:

   - Input patient data
   - Make predictions
   - Train new models
   - View risk factors

3. For first-time use:
   - The system will automatically train a new model if none exists
   - You can also click "Train New Model" to retrain at any time

## Directory Structure

```
.
├── data
│   └── processed
│       └── heart.csv
├── models
│   └── heart_disease_model.joblib
├── heart_disease_system.py
├── requirements.txt
└── README.md
```

## Project Structure

```
heart_Attack_prediction/
│
├── data/               # Data files
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
│
├── docs/              # Documentation
│   ├── data_dict.md   # Data dictionary
│   └── model_eval.md  # Model evaluation details
│
├── models/            # Trained models
│   └── heart_disease_model.joblib
│
├── results/           # Results and outputs
│   └── plots/         # Visualization plots
│
├── src/              # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model training scripts
│   ├── visualization/# Visualization scripts
│   └── gui/          # GUI application
│
├── tests/            # Test files
│
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Model Performance

The system implements multiple machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Naive Bayes
- K-Nearest Neighbors

Best performing model: Gradient Boosting

- Accuracy: 91.67%
- Precision: 92.16%
- Recall: 97.92%
- F1-Score: 94.95%

## Data Features

The model uses the following clinical parameters:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression
- ST Slope

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
