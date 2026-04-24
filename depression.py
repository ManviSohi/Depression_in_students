import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# 1. Dataset Load karna
df = pd.read_csv('Student Depression Dataset.csv')

# 2. Data Cleaning & Preprocessing
# 'id' column ki zaroorat nahi hai, isliye drop kar rahe hain
df = df.drop(columns=['id'])

# Missing values handle karna (Rows with NaN remove kar rahe hain)
df = df.dropna()

# Categorical data (text) ko numbers mein convert karna (Encoding)
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Classification Task (Target: Depression - Yes/No)
X_clf = df.drop(columns=['Depression'])
y_clf = df['Depression']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Models Train karna
log_model = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

log_model.fit(X_train_c, y_train_c)
rf_clf.fit(X_train_c, y_train_c)

acc_log = accuracy_score(y_test_c, log_model.predict(X_test_c))
acc_rf = accuracy_score(y_test_c, rf_clf.predict(X_test_c))

# 4. Regression Task (Target: CGPA)
X_reg = df.drop(columns=['CGPA'])
y_reg = df['CGPA']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Models Train karna
lin_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

lin_reg.fit(X_train_r, y_train_r)
rf_reg.fit(X_train_r, y_train_r)

r2_lin = r2_score(y_test_r, lin_reg.predict(X_test_r))
r2_rf = r2_score(y_test_r, rf_reg.predict(X_test_r))

# 5. Result Visualization
labels = ['LogReg (Acc)', 'RF Clf (Acc)', 'LinReg (R2)', 'RF Reg (R2)']
scores = [acc_log, acc_rf, r2_lin, r2_rf]

plt.figure(figsize=(10, 6))
sns.barplot(x=labels, y=scores, palette='viridis')
plt.title('Performance Comparison: Classification vs Regression')
plt.ylabel('Score (Higher is better for Acc, Close to 1 for R2)')
plt.show()

print(f"Classification Accuracy (LogReg): {acc_log:.2f}")
print(f"Classification Accuracy (Random Forest): {acc_rf:.2f}")
print(f"Regression R2 Score (LinReg): {r2_lin:.2f}")
print(f"Regression R2 Score (Random Forest): {r2_rf:.2f}")
