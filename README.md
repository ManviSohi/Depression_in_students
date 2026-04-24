# Depression_in_students
# Student Depression & Academic Performance Analysis

This project explores the 'Student Depression Dataset' using Machine Learning to perform two distinct tasks:
1. **Classification**: Predicting the presence of Depression (Binary: Yes/No).
2. **Regression**: Estimating academic performance (CGPA) based on lifestyle and psychological factors.

## 📊 Dataset Overview
The dataset contains information about students including:
- **Demographics**: Gender, Age, City.
- **Psychological Factors**: Academic Pressure, Study Satisfaction, Suicidal Thoughts.
- **Lifestyle**: Sleep Duration, Dietary Habits, Work/Study Hours.
- **Target Variables**: `Depression` (for Classification) and `CGPA` (for Regression).

## 🚀 Machine Learning Models

### Classification (Target: Depression)
- **Logistic Regression**: Used as a baseline linear classifier.
- **Random Forest Classifier**: Used to capture non-linear relationships between academic pressure and mental health.
- **Result**: Achieved high accuracy (~84%), indicating strong correlation between input factors and depression levels.

### Regression (Target: CGPA)
- **Linear Regression**: To model the linear impact of study hours and sleep on grades.
- **Random Forest Regressor**: To identify complex patterns in academic outcomes.
- **Result**: Low R² score, suggesting that CGPA is influenced by factors beyond those captured linearly in this specific dataset.

## 📈 Key Findings & Visualizations
- A comparative bar chart (`performance_comparison.png`) was generated to show the performance gap between Classification (Accuracy) and Regression (R² Score).
- Categorical features like 'Sleep Duration' and 'Dietary Habits' were encoded to improve model learning.

## 🛠️ Requirements
- Python 3.x
- Pandas, Scikit-learn, Matplotlib, Seaborn
