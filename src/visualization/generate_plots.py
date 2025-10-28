import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_plots(data_path='../../data/processed/heart.csv', output_dir='../../results/plots'):
    """Generate visualization plots for the heart disease dataset."""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style for better-looking plots
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # 1. Data Distribution Summary
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='HeartDisease')
    plt.title('Heart Disease Distribution')
    
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='Age', bins=20)
    plt.title('Age Distribution')
    
    plt.subplot(2, 2, 3)
    sns.countplot(data=df, x='Sex', hue='HeartDisease')
    plt.title('Heart Disease by Gender')
    
    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='ChestPainType', hue='HeartDisease')
    plt.title('Heart Disease by Chest Pain Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    df_corr = df.copy()
    df_corr['Sex'] = df_corr['Sex'].map({'M': 1, 'F': 0})
    df_corr['ChestPainType'] = df_corr['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df_corr['RestingECG'] = df_corr['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df_corr['ExerciseAngina'] = df_corr['ExerciseAngina'].map({'Y': 1, 'N': 0})
    df_corr['ST_Slope'] = df_corr['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, '2_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Clinical Measurements
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='HeartDisease', y='RestingBP')
    plt.title('Resting BP Distribution')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(data=df, x='HeartDisease', y='Cholesterol')
    plt.title('Cholesterol Distribution')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='HeartDisease', y='MaxHR')
    plt.title('Max Heart Rate Distribution')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='HeartDisease', y='Oldpeak')
    plt.title('ST Depression (Oldpeak) Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_clinical_measurements.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Age-related Analysis
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=df, x='Age', y='RestingBP', hue='HeartDisease')
    plt.title('Age vs Resting BP')
    
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='Age', y='Cholesterol', hue='HeartDisease')
    plt.title('Age vs Cholesterol')
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='Age', y='MaxHR', hue='HeartDisease')
    plt.title('Age vs Max Heart Rate')
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df, x='Age', y='Oldpeak', hue='HeartDisease')
    plt.title('Age vs ST Depression')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_age_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Model Performance Comparison
    models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'Naive Bayes', 'KNN']
    accuracy = [0.8000, 0.8500, 0.9167, 0.8167, 0.8333, 0.8667]
    precision = [0.8462, 0.8545, 0.9216, 0.8364, 0.8519, 0.8704]
    recall = [0.9167, 0.9792, 0.9792, 0.9583, 0.9583, 0.9792]
    f1 = [0.8800, 0.9126, 0.9495, 0.8932, 0.9020, 0.9216]
    
    plt.figure(figsize=(15, 8))
    x = np.arange(len(models))
    width = 0.2
    
    plt.bar(x - width*1.5, accuracy, width, label='Accuracy', color='skyblue')
    plt.bar(x - width/2, precision, width, label='Precision', color='lightgreen')
    plt.bar(x + width/2, recall, width, label='Recall', color='salmon')
    plt.bar(x + width*1.5, f1, width, label='F1-Score', color='purple')
    
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_model_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Feature Importance
    features = ['Age', 'Oldpeak', 'Cholesterol', 'ChestPainType', 'RestingBP', 
               'ExerciseAngina', 'Sex', 'MaxHR', 'ST_Slope', 'FastingBS']
    importance = [0.190237, 0.183559, 0.133791, 0.129391, 0.106380,
                 0.104762, 0.067603, 0.041779, 0.040589, 0.001908]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importance, y=features)
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots have been generated and saved in '{output_dir}'")
    print("\nGenerated files:")
    for file in sorted(os.listdir(output_dir)):
        print(f"- {file}")

if __name__ == "__main__":
    create_plots() 