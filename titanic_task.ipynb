{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9484bde9-024d-4127-ac40-0c88cd72426f",
   "metadata": {},
   "source": [
    "# Titanic - Kaggle Project"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df63339d-6f7a-4179-ae5c-a6cb540d88a3",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "##  Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "919d750f-54d3-40d1-8f9b-896400a55bf8",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.impute import KNNImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "309c82d2-8dab-49d4-8759-b78bef7cbcc2",
   "metadata": {},
   "source": [
    "## 1. Data Preprocessing Helpers\n",
    "### Helper functions:\n",
    "\n",
    "#### handle_missing_values\n",
    "\n",
    "#### encode_categorical\n",
    "\n",
    "#### scale_numeric\n",
    "\n",
    "#### drop_unnecessary\n",
    "\n",
    "#### handle_outliers\n",
    "\n",
    "#### preprocess_data\n",
    "\n",
    "#### split_dat"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7a68cad5-e012-4c59-b625-fe4f89e9274a",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "def handle_missing_values(df):\n",
    "    \"\"\"Handle missing values: Cabin -> extract letter, Embarked -> fill with mode, Age -> imputation\"\"\"\n",
    "    df_processed = df.copy()\n",
    "\n",
    "    # Cabin -> Extract first letter, drop original\n",
    "    if 'Cabin' in df_processed.columns:\n",
    "        df_processed['Cabin_Letter'] = df_processed['Cabin'].astype(str).str[0]\n",
    "        df_processed = df_processed.drop('Cabin', axis=1)\n",
    "\n",
    "    # Embarked -> Fill with mode\n",
    "    if 'Embarked' in df_processed.columns:\n",
    "        embarked_mode = df_processed['Embarked'].mode()[0]\n",
    "        df_processed['Embarked'] = df_processed['Embarked'].fillna(embarked_mode)\n",
    "\n",
    "    # Age -> Imputation\n",
    "    if 'Age' in df_processed.columns:\n",
    "        # Group median by Pclass & Sex\n",
    "        age_median_by_group = df_processed.groupby(['Pclass', 'Sex'])['Age'].median()\n",
    "        df_processed['Age'] = df_processed.apply(\n",
    "            lambda row: age_median_by_group[(row['Pclass'], row['Sex'])] \n",
    "            if pd.isna(row['Age']) else row['Age'], axis=1\n",
    "        )\n",
    "\n",
    "        # KNN Imputation for numerical columns\n",
    "        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()\n",
    "        for col in ['Survived', 'PassengerId']:\n",
    "            if col in numeric_cols:\n",
    "                numeric_cols.remove(col)\n",
    "        if numeric_cols:\n",
    "            knn_imputer = KNNImputer(n_neighbors=5)\n",
    "            df_processed[numeric_cols] = knn_imputer.fit_transform(df_processed[numeric_cols])\n",
    "\n",
    "    return df_processed\n",
    "\n",
    "\n",
    "def encode_categorical(df):\n",
    "    \"\"\"One-Hot Encoding for categorical columns\"\"\"\n",
    "    categorical_cols = ['Sex', 'Embarked', 'Cabin_Letter']\n",
    "    categorical_cols = [c for c in categorical_cols if c in df.columns]\n",
    "    if categorical_cols:\n",
    "        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
    "    return df\n",
    "\n",
    "\n",
    "def scale_numeric(df):\n",
    "    \"\"\"Standardize numeric columns\"\"\"\n",
    "    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
    "    if 'Survived' in numeric_cols:\n",
    "        numeric_cols.remove('Survived')\n",
    "    if 'PassengerId' in numeric_cols:\n",
    "        numeric_cols.remove('PassengerId')\n",
    "    if numeric_cols:\n",
    "        scaler = StandardScaler()\n",
    "        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])\n",
    "    return df\n",
    "\n",
    "\n",
    "def drop_unnecessary(df, cols):\n",
    "    \"\"\"Drop unnecessary columns if exist\"\"\"\n",
    "    cols = [c for c in cols if c in df.columns]\n",
    "    return df.drop(columns=cols)\n",
    "\n",
    "\n",
    "def preprocess_data(df, is_train=True):\n",
    "    \"\"\"Full preprocessing pipeline\"\"\"\n",
    "    df = handle_missing_values(df)\n",
    "    df = encode_categorical(df)\n",
    "    df = scale_numeric(df)\n",
    "    df = drop_unnecessary(df, ['PassengerId', 'Name', 'Ticket'])\n",
    "    return df\n",
    "\n",
    "\n",
    "def split_data(df, target_column='Survived'):\n",
    "    \"\"\"Split dataframe into train/test sets\"\"\"\n",
    "    X = df.drop(columns=[target_column])\n",
    "    y = df[target_column]\n",
    "    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e0b7a320-9a78-4d6a-a2f7-cbaafb0bb469",
   "metadata": {},
   "source": [
    "## 2. Load and Preprocess Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "38e94894-905a-4b76-9c85-5d3b4cf8af37",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = pd.read_csv('Desktop/coding/AI exercise/ML_titanic_task/train.csv')\n",
    "df_test = pd.read_csv('Desktop/coding/AI exercise/ML_titanic_task/test.csv')\n",
    "\n",
    "df_train_clean = preprocess_data(df_train, is_train=True)\n",
    "X_train, X_val, y_train, y_val = split_data(df_train_clean, target_column='Survived')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "91443527-6fe4-4a9a-8422-632376cc7f20",
   "metadata": {},
   "source": [
    "## 3. Model Training\n",
    "\n",
    "### Model training and evaluation:\n",
    "\n",
    "#### Decision Tree & Random Forest\n",
    "\n",
    "#### Cross Validation\n",
    "\n",
    "#### Grid Search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e65fc319-71cc-484a-af34-e42851f01ad9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 0.7933\n",
      "Random Forest Accuracy: 0.8045\n",
      "Cross-Validation Accuracies: [0.77653631 0.78651685 0.84269663 0.74719101 0.83707865]\n",
      "Mean CV Accuracy: 0.7980\n",
      "Best Params: {'max_depth': 5, 'n_estimators': 200}\n",
      "Best CV Accuracy: 0.8306\n"
     ]
    }
   ],
   "source": [
    "# Decision Tree\n",
    "dt_model = DecisionTreeClassifier(random_state=42)\n",
    "dt_model.fit(X_train, y_train)\n",
    "dt_preds = dt_model.predict(X_val)\n",
    "print(f\"Decision Tree Accuracy: {accuracy_score(y_val, dt_preds):.4f}\")\n",
    "\n",
    "# Random Forest\n",
    "rf_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "rf_model.fit(X_train, y_train)\n",
    "rf_preds = rf_model.predict(X_val)\n",
    "print(f\"Random Forest Accuracy: {accuracy_score(y_val, rf_preds):.4f}\")\n",
    "\n",
    "# Cross-validation\n",
    "cv_scores = cross_val_score(rf_model, df_train_clean.drop(columns=['Survived']), df_train_clean['Survived'], cv=5)\n",
    "print(f\"Cross-Validation Accuracies: {cv_scores}\")\n",
    "print(f\"Mean CV Accuracy: {cv_scores.mean():.4f}\")\n",
    "\n",
    "# Grid Search\n",
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'max_depth': [None, 5, 10]\n",
    "}\n",
    "grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)\n",
    "grid_search.fit(df_train_clean.drop(columns=['Survived']), df_train_clean['Survived'])\n",
    "print(f\"Best Params: {grid_search.best_params_}\")\n",
    "print(f\"Best CV Accuracy: {grid_search.best_score_:.4f}\")\n",
    "\n",
    "best_model = grid_search.best_estimator_"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13a19a7f-2b94-47c5-a4c5-db3e17aa4ed1",
   "metadata": {},
   "source": [
    "## 4. Prediction & Submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2b0c5fc6-b928-4086-adb4-0a38ddf42dbc",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ submission.csv created successfully\n"
     ]
    }
   ],
   "source": [
    "df_test_clean = preprocess_data(df_test, is_train=False)\n",
    "\n",
    "# Align columns with training set\n",
    "df_test_clean = df_test_clean.reindex(columns=df_train_clean.drop(columns=['Survived']).columns, fill_value=0)\n",
    "\n",
    "# Predict\n",
    "predictions = best_model.predict(df_test_clean)\n",
    "\n",
    "# Create submission file\n",
    "submission = pd.DataFrame({\n",
    "    \"PassengerId\": df_test[\"PassengerId\"],\n",
    "    \"Survived\": predictions\n",
    "})\n",
    "submission.to_csv(\"submission.csv\", index=False)\n",
    "print(\"✅ submission.csv created successfully\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
