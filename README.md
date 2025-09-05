 # Titanic Survival Prediction - Kaggle Project

## 🎯 Project Goal
The goal of this project is to build a Machine Learning model to predict whether a passenger survived the Titanic disaster.  
We use the **Kaggle Titanic dataset**, train models on the training set, and generate predictions for the test set.  

---

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected dataset structure with `.info()` and `.head()`.
- Checked missing values using `isnull().sum()`.
- Found missing data in `Age`, `Cabin`, and `Embarked`.

---

### 2. Data Preprocessing
#### 2.1 Handling Missing Values
- **Cabin** → Dropped original column but extracted the first letter (`Cabin_Letter`).
- **Embarked** → Filled missing values with the mode.
- **Age** → 
  - Filled by group median (`Pclass`, `Sex`).
  - Then refined with **KNN Imputation** for better accuracy.

#### 2.2 Encoding Categorical Features
- Converted `Sex`, `Embarked`, and `Cabin_Letter` into numeric form using **One-Hot Encoding**.

#### 2.3 Feature Scaling
- Standardized numerical columns (`Age`, `Fare`, etc.) with **StandardScaler**.

#### 2.4 Dropping Unnecessary Columns
- Removed `PassengerId`, `Name`, `Ticket` since they don’t help prediction.

---

### 3. Splitting Data
- Training data split into:
  - **80% train** (for model fitting)
  - **20% validation** (for performance check)

---

### 4. Modeling
- **Decision Tree Classifier** → Accuracy ~79.3%
- **Random Forest Classifier** → Accuracy ~80.4%

---

### 5. Model Validation
- Used **5-Fold Cross Validation** to check stability.
- Mean CV accuracy ~79.8%.

---

### 6. Hyperparameter Tuning
- Applied **GridSearchCV** to optimize Random Forest.
- Best parameters:  
  - `max_depth=5`
  - `n_estimators=200`
- Improved CV accuracy to ~83.0%.

---

### 7. Predictions on Test Set
- Preprocessed test data with the same pipeline.
- Ensured train/test columns match (`reindex` with missing columns filled as 0).
- Generated predictions with the best model.
- Saved results as `submission.csv`.

---

## 📊 Results
- Decision Tree Accuracy: **0.7933**
- Random Forest Accuracy: **0.8045**
- Cross-Validation Mean Accuracy: **0.7980**
- Best Model (Random Forest with tuned params): **0.8306 CV Accuracy**
- Final submission file: **submission.csv**

---

## 📈 Learnings
1. End-to-End ML workflow (EDA → Preprocessing → Modeling → Submission).
2. Handling missing data with group-based medians and KNN imputation.
3. Encoding categorical features via One-Hot Encoding.
4. Feature scaling for numerical stability.
5. Hyperparameter tuning with GridSearchCV.
6. Ensuring train/test feature alignment before prediction.
7. Understanding Kaggle workflow: model → predictions → submission file.

---

## ✅ Conclusion
We successfully built and tuned a Random Forest model that achieved **~83% accuracy** with cross-validation.  
The project provided practical experience in **data preprocessing, feature engineering, model selection, hyperparameter tuning, and Kaggle submission workflow**.
