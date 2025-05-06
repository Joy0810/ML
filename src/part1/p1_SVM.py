import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Label Encoding Attrition
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes -> 1, No -> 0

# Optional: Visualize class imbalance
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Class Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and labels
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Classifier with class_weight
model = SVC(class_weight='balanced', probability=True, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.set(style='whitegrid')

# Create figure
plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    linewidths=0.5,
    linecolor='gray',
    square=True,
    xticklabels=['No', 'Yes'],
    yticklabels=['No', 'Yes'],
    annot_kws={"size": 14, "weight": "bold"}
)

# Formatting
ax.set_title('Confusion Matrix – SVM', fontsize=16, weight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('Actual Label', fontsize=12)

# Move x-axis label and ticks to top
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# Increase font size of tick labels
ax.tick_params(axis='both', labelsize=12)

# Tight layout for report-quality spacing
plt.tight_layout()
plt.show()
# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title('ROC Curve - SVM')
plt.show()
