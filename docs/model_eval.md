# Model Evaluation Documentation

## Model Performance Summary

### Best Model: Gradient Boosting Classifier

#### Performance Metrics

- Accuracy: 91.67%
- Precision: 92.16%
- Recall: 97.92%
- F1-Score: 94.95%
- AUC-ROC: 0.9809

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1-Score | AUC    |
| ------------------- | -------- | --------- | ------ | -------- | ------ |
| Gradient Boosting   | 0.9167   | 0.9216    | 0.9792 | 0.9495   | 0.9809 |
| Random Forest       | 0.8500   | 0.8545    | 0.9792 | 0.9126   | 0.9644 |
| K-Nearest Neighbors | 0.8667   | 0.8704    | 0.9792 | 0.9216   | 0.7188 |
| Naive Bayes         | 0.8333   | 0.8519    | 0.9583 | 0.9020   | 0.8056 |
| SVM                 | 0.8167   | 0.8364    | 0.9583 | 0.8932   | 0.8108 |
| Logistic Regression | 0.8000   | 0.8462    | 0.9167 | 0.8800   | 0.8733 |

## Feature Importance

### Top Features by Importance Score

1. Age (0.190237)
2. ST Depression/Oldpeak (0.183559)
3. Cholesterol (0.133791)
4. Chest Pain Type (0.129391)
5. Resting BP (0.106380)
6. Exercise Angina (0.104762)
7. Sex (0.067603)
8. Max Heart Rate (0.041779)
9. ST Slope (0.040589)
10. Fasting Blood Sugar (0.001908)

## Model Training Details

### Data Split

- Training set: 80% (240 samples)
- Test set: 20% (60 samples)
- Stratification: Yes (based on target variable)

### Preprocessing Steps

1. Categorical variable encoding
2. Feature scaling (StandardScaler)
3. Feature selection (SelectKBest)
4. Missing value handling
5. Outlier treatment

### Cross-validation Results

- Method: 5-fold cross-validation
- Mean CV Score: 0.8708
- Standard Deviation: 0.0667

## Risk Level Classification

### Probability Thresholds

- Low Risk: < 0.3
- Moderate Risk: 0.3 - 0.7
- High Risk: > 0.7

### Risk Factors Weight

1. Age > 55 years
2. Male gender
3. Asymptomatic chest pain
4. High blood pressure (>140 mmHg)
5. High cholesterol (>240 mg/dl)
6. Exercise-induced angina
7. High ST depression (>2)
8. Downsloping ST segment
9. Low maximum heart rate
10. Abnormal resting ECG

## Model Limitations and Considerations

### Known Limitations

1. Dataset size (300 samples)
2. Demographic representation
3. Feature correlation effects
4. Model interpretability challenges

### Usage Guidelines

1. Use as support tool, not sole decision maker
2. Consider confidence scores
3. Regular model retraining recommended
4. Monitor for drift in predictions

## Future Improvements

### Planned Enhancements

1. Larger training dataset
2. Additional feature engineering
3. Deep learning implementation
4. Ensemble method optimization
5. Real-time monitoring system

### Validation Strategy

1. External validation dataset
2. Clinical trial validation
3. Continuous performance monitoring
4. Regular model updates
