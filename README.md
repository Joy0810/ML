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

### Model Performance (Part 1)
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

 ---

# Multi-Step Regression + Classification for Employee Attrition & Salary Estimation

## Part 2: Simulating Future Salaries (Data Augmentation)

---

## Objective
The goal of Part 2 is to simulate future salary predictions for employees based on two growth models:
1. **Fixed Growth Increment**: A uniform salary increase (8%) applied to all employees.
2. **Performance-Based Growth Increment**: Salary increase based on an employee’s performance rating.

Additionally, a **Linear Regression** model is used to analyze the relationship between an employee’s **Performance Rating** and their **simulated future salary**.

---

## Dataset
- **File Path:** `dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Features:** Includes attributes like `MonthlyIncome`, `PerformanceRating`, etc.
- **Target Column:** Simulated future salaries based on fixed growth and performance-based increments.

---

## Simulation Steps
1. **Fixed Growth Increment**: 
   - An 8% increase is applied uniformly to all employees’ monthly income, with added noise (±3%) for realistic simulation.
   
2. **Performance-Based Growth Increment**:
   - Employees with a performance rating of 4 receive a 10% increase, and those with lower ratings receive a 5% increase. Noise is also added to this model.

3. **Linear Regression**:
   - A **Linear Regression** model is built to explore the relationship between **Performance Rating** and **simulated future salary**.

---

## Simulation Results
- Both **Fixed Growth** and **Performance-Based Growth** models produce future salary predictions, which are saved in a new dataset for further analysis.
- The **linear regression model** reveals how performance ratings influence salary increments in the performance-based model.

---

## Model Performance (Part 2)
- The simulation provides future salary predictions based on both growth models. **Performance-Based Growth** is more dynamic as it accounts for individual performance ratings, while the **Fixed Growth Increment** offers a simpler, uniform approach.
- The **Linear Regression** model successfully identifies the correlation between **Performance Rating** and the **future salary** prediction under the performance-based model.

---

###  Folder Structure Overview

```
hr/
│
├── dataset/
│ └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── images/
│ └── part1/
│ ├── (Sub-folders covering images for Decision_tree, Logistic_regression, Random_forest, SVM, XG_Boost)
│ ├── Class_importance.png
│ ├── part1_main.png    (Logistic Regression performs the best)
│ └── part1_XG.png    (Without Logistic regression XG Boost performs better)
│ └── part2/
│ ├── salary_distribution.png (KDE plot comparing salary distributions)
│ └── part3/
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
│ └── part1XG_output.txt   (XGBoost Output)
│ └── part2/
│ ├── augmented_salary_data.csv     (Augmented dataset with future salary predictions)
│ └── p2.py     (Simulate Future Salaries)

```
