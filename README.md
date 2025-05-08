#  Multi-Step Regression + Classification for Employee Attrition & Salary Estimation

## Part 1: Employee Attrition Prediction (Classification)

---

##  Objective
The goal of Part 1 is to build and evaluate multiple classification models to predict whether an employee will leave the company (Attrition = Yes/No) using IBM's HR Analytics dataset.

---

##  Dataset
- **File Path:** `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Features:** Includes attributes like `Age`, `BusinessTravel`, `Department`, `Education`, `EnvironmentSatisfaction`, `JobRole`, `MonthlyIncome`, etc.
- **Target Column:** `Attrition` (Binary: Yes/No)

---

##  Preprocessing Steps
- Handling categorical variables using Label Encoding
- Scaling numerical features using `StandardScaler`
- Train-test split (80% train, 20% test)
- Handled class imbalance using SMOTE in some models

---

##  Classification Models Used

### 1. Logistic Regression
- Simple, interpretable baseline model for binary classification.

### 2. Decision Tree
- Tree-based model for capturing feature interactions.

### 3. Random Forest
- An ensemble of decision trees to reduce overfitting and improve accuracy.

### 4. Random Forest + SMOTE
- Used SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance before applying RF.

### 5. XGBoost
- Gradient boosting algorithm that provides high performance on structured data.

### 6. Support Vector Machine (SVM)
- Maximizes class margin in high-dimensional space; good for binary classification.

---

### Model Performance
- Both **Logistic Regression** and **XGBoost** showed the best performance in predicting employee attrition. However, **Logistic Regression** outperformed **XGBoost** by a slight margin, making it the more reliable model for this classification task.

---

##  Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC-AUC Curve**

---

##  Folder Structure Overview

```
hr/
│
├── dataset/
│ └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── images/
│ └── part1/
│ └── [confusion matrices, ROC curves, metric plots]
│
├── src/
│ └── part1/
│ ├── evaluation.py   (To evaluate)
│ ├── p1_DT.py   (Decision Tree)
│ ├── p1_LG.py   (Logistic Regression)
│ ├── p1_RF.py   (Random Forest)
│ ├── p1_RF_smt.py   (Random Forest + SMOTE)
│ ├── p1_SVM.py   (SVM)
│ ├── p1_XG.py   (XGBoost)
│ ├── part1lg_output.txt   (Logistic Regression Output)
│ └── part1XG_output.txt   (XGBoost 
```
## 📦 Python Libraries Used (For Part 1)

- **Data Manipulation**: 
    - `pandas`
    - `numpy`

- **Visualization**:
    - `matplotlib`
    - `seaborn`

- **Machine Learning**:
    - `scikit-learn`
        - `LogisticRegression`
        - `DecisionTreeClassifier`
        - `RandomForestClassifier`
        - `SVC`
        - `classification_report`
        - `confusion_matrix`
        - `train_test_split`
        - `roc_auc_score`
        - `accuracy_score`
        - `f1_score`
        - `precision_score`
        - `recall_score`

- **Handling Imbalance**:
    - `imblearn` (SMOTE)

- **Boosting**:
    - `xgboost`
