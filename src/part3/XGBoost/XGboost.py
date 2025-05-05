import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('augmented_salary_data.csv')

# Drop rows with NaNs (if any)
df = df.dropna()

# Common features for both models
features = ['Age', 'Education', 'Experience', 'MonthlyIncome', 'Rating', 'PerformanceScore']

# -----------------------------------------------
# Model A: Predict FutureSalary_PerformanceBased
# -----------------------------------------------
X_A = df[features]
y_A = df['FutureSalary_PerformanceBased']

X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_A, test_size=0.2, random_state=42)

xgb_A = XGBRegressor(random_state=42)
xgb_A.fit(X_train_A, y_train_A)

y_pred_A = xgb_A.predict(X_test_A)

# Metrics
r2_A = r2_score(y_test_A, y_pred_A)
rmse_A = np.sqrt(mean_squared_error(y_test_A, y_pred_A))
mape_A = mean_absolute_percentage_error(y_test_A, y_pred_A) * 100

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_A, y=y_pred_A, color='blue', s=40)
plt.plot([y_test_A.min(), y_test_A.max()], [y_test_A.min(), y_test_A.max()], 'r--', lw=2)
plt.title(f"XGBoost Model A: Predicting Performance-Based Salary\nR²: {r2_A:.4f} | RMSE: {rmse_A:.2f} | MAPE: {mape_A:.2f}%", fontsize=11)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.tight_layout()
plt.savefig('/mnt/data/xgb_performance_based_actual_vs_predicted.png')
plt.close()

# -----------------------------------------------
# Model B: Predict Increment Rate → Multiply with MonthlyIncome
# -----------------------------------------------
df['IncrementRate'] = df['FutureSalary_FixedGrowth'] / df['MonthlyIncome']

X_B = df[features]
y_B = df['IncrementRate']

X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X_B, y_B, test_size=0.2, random_state=42)

xgb_B = XGBRegressor(random_state=42)
xgb_B.fit(X_train_B, y_train_B)

y_pred_B = xgb_B.predict(X_test_B)

# Multiply back with MonthlyIncome to get predicted salary
monthly_income_B = X_test_B['MonthlyIncome'].values
y_pred_salary_B = y_pred_B * monthly_income_B
y_actual_salary_B = y_test_B * monthly_income_B

# Metrics
r2_B = r2_score(y_actual_salary_B, y_pred_salary_B)
rmse_B = np.sqrt(mean_squared_error(y_actual_salary_B, y_pred_salary_B))
mape_B = mean_absolute_percentage_error(y_actual_salary_B, y_pred_salary_B) * 100

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_actual_salary_B, y=y_pred_salary_B, color='green', s=40)
plt.plot([y_actual_salary_B.min(), y_actual_salary_B.max()], [y_actual_salary_B.min(), y_actual_salary_B.max()], 'r--', lw=2)
plt.title(f"XGBoost Model B: Predicting Increment → Salary\nR²: {r2_B:.4f} | RMSE: {rmse_B:.2f} | MAPE: {mape_B:.2f}%", fontsize=11)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.tight_layout()
plt.savefig('/mnt/data/xgb_increment_based_actual_vs_predicted.png')
plt.close()

# -----------------------------------------------
# Print final comparison
# -----------------------------------------------
print("🔍 XGBoost Results:")
print("\nModel A: Predict Performance-Based Salary")
print(f"R² Score: {r2_A:.4f}")
print(f"RMSE    : {rmse_A:.2f}")
print(f"MAPE    : {mape_A:.2f}%")

print("\nModel B: Predict Increment Rate then Multiply")
print(f"R² Score: {r2_B:.4f}")
print(f"RMSE    : {rmse_B:.2f}")
print(f"MAPE    : {mape_B:.2f}%")