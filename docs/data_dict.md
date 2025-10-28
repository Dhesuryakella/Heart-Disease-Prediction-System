# Heart Disease Dataset Documentation

## Dataset Overview

This dataset is used for predicting heart disease based on various medical attributes. It contains both categorical and numerical features that are commonly used in cardiovascular health assessment.

## Features Description

### Demographic Information

1. **Age**

   - Type: Numerical
   - Range: 40-75 years
   - Description: Age of the patient

2. **Sex**
   - Type: Categorical
   - Values:
     - M: Male
     - F: Female
   - Description: Gender of the patient

### Clinical Features

3. **ChestPainType**

   - Type: Categorical
   - Values:
     - ASY: Asymptomatic
     - ATA: Atypical Angina
     - NAP: Non-Anginal Pain
     - TA: Typical Angina
   - Description: Type of chest pain experienced by the patient
   - Note: ASY (Asymptomatic) often indicates higher risk

4. **RestingBP**

   - Type: Numerical
   - Range: 90-200 mmHg
   - Description: Resting blood pressure
   - Clinical significance:
     - Normal: < 120 mmHg
     - Elevated: 120-129 mmHg
     - High (Stage 1): 130-139 mmHg
     - High (Stage 2): ≥ 140 mmHg

5. **Cholesterol**

   - Type: Numerical
   - Range: 150-400 mg/dl
   - Description: Serum cholesterol level
   - Clinical significance:
     - Normal: < 200 mg/dl
     - Borderline high: 200-239 mg/dl
     - High: ≥ 240 mg/dl

6. **FastingBS**

   - Type: Categorical (Binary)
   - Values:
     - 0: Fasting blood sugar < 120 mg/dl
     - 1: Fasting blood sugar > 120 mg/dl
   - Description: Fasting blood sugar level
   - Note: Values > 120 mg/dl may indicate diabetes

7. **RestingECG**

   - Type: Categorical
   - Values:
     - Normal: Normal ECG
     - ST: Having ST-T wave abnormality
     - LVH: Showing probable or definite left ventricular hypertrophy
   - Description: Resting electrocardiogram results

8. **MaxHR**

   - Type: Numerical
   - Range: 60-220 bpm
   - Description: Maximum heart rate achieved
   - Formula: Generally decreases with age
   - Typical max HR = 220 - age

9. **ExerciseAngina**

   - Type: Categorical
   - Values:
     - Y: Yes
     - N: No
   - Description: Exercise-induced angina
   - Note: Presence (Y) indicates higher risk

10. **Oldpeak**

    - Type: Numerical
    - Range: 0-6.0
    - Description: ST depression induced by exercise relative to rest
    - Clinical significance:
      - Higher values indicate greater likelihood of heart disease

11. **ST_Slope**
    - Type: Categorical
    - Values:
      - Up: Upsloping
      - Flat: Flat
      - Down: Downsloping
    - Description: Slope of the peak exercise ST segment
    - Clinical significance:
      - Downsloping is associated with higher risk
      - Upsloping is generally considered normal

### Target Variable

12. **HeartDisease**
    - Type: Binary
    - Values:
      - 0: No heart disease
      - 1: Heart disease
    - Description: Presence of heart disease

## Data Collection Guidelines

1. **Measurement Conditions**:

   - All measurements should be taken when the patient is at rest
   - ECG readings should be properly calibrated
   - Blood pressure should be measured in a seated position

2. **Quality Control**:

   - Ensure all measurements are within valid ranges
   - Verify categorical values match expected options
   - Document any unusual or outlier values

3. **Missing Values**:
   - Document reason for missing values
   - Use appropriate imputation methods
   - Flag records with imputed values
