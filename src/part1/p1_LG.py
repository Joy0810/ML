import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay

df = pd.read_csv("../../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

sns.countplot(x='Attrition', data=df)
plt.title("Attrition Class Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", round(roc_auc_score(y_test, y_proba_test), 4))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.set(style='whitegrid')
plt.figure(figsize=(6, 5))
ax = sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray', square=True, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], annot_kws={"size": 14, "weight": "bold"})
ax.set_title('Confusion Matrix – Logistic Regression', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve - Logistic Regression')
plt.show()

X_all_scaled = scaler.transform(X)
y_proba_all = model.predict_proba(X_all_scaled)[:, 1]

output_df = pd.DataFrame({"EmployeeIndex": X.index, "Attrition_Probability": y_proba_all})
output_df.to_csv("part1lg_output.txt", index=False, sep='\t')
print("Predicted attrition probabilities saved to part1lg_output.txt")
