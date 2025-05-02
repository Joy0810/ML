import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("../../dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Label Encoding Attrition
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes -> 1, No -> 0

# Visualize class imbalance
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Class Distribution")
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.show()

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# Split features and labels
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Grid Search for Random Forest hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1)

# Fit grid search on the SMOTE-applied training data
grid_search.fit(X_train_smote, y_train_smote)

# Best parameters and model
print("Best Parameters:\n", grid_search.best_params_)

# Use best estimator from grid search
best_rf = grid_search.best_estimator_

# Predictions
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title('Confusion Matrix - Random Forest + SMOTE + GridSearch')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(best_rf, X_test, y_test)
plt.title('ROC Curve - Random Forest + SMOTE + GridSearch')
plt.show()

# Feature Importance
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Important Features - Random Forest + SMOTE + GridSearch")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
