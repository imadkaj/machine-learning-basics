# 🛳️ Titanic Survival Prediction using KNN (scikit-learn)

This project is a simple machine learning pipeline that predicts whether a Titanic passenger survived based on key features such as gender, class, and fare. The model uses **K-Nearest Neighbors (KNN)** and is built with **Python** and **scikit-learn**.

---

## 📌 Project Overview

- 📂 Dataset: [`titanic.csv`](https://www.kaggle.com/competitions/titanic/data)
- 📈 Model: K-Nearest Neighbors Classifier
- 🛠 Tools: pandas, scikit-learn, seaborn, matplotlib

The model is trained to classify survival based on passenger features after thorough data cleaning and feature engineering.

---

## 🚀 Features

### ✅ Data Cleaning
- Dropped unnecessary columns: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Filled missing values in `Embarked` and `Age`
- Dropped `Embarked` column after fill

### 🧠 Feature Engineering
- Converted `Sex` to numeric values (0 = female, 1 = male)
- Created new features:
  - `FamilySize` (sum of siblings/spouse and parents/children aboard)
  - `IsAlone` (1 if traveling alone, 0 otherwise)
  - `FareBin` and `AgeBin` (binned fare and age into categories)

### ⚙️ Model Training
- Used `MinMaxScaler` to normalize numerical features
- Tuned KNN model using `GridSearchCV` with 5-fold cross-validation
- Evaluated using accuracy and confusion matrix

### 📊 Visualization
- Displays a heatmap of the confusion matrix using `seaborn`

---

## 📁 File Structure

```bash
├── main.py             # Main Python script with full ML pipeline
├── titanic.csv         # Titanic dataset (from Kaggle)
├── README.md           # Project documentation

