# Titanic Survival Prediction – Machine Learning Project

## Objective
To predict whether a passenger survived the Titanic disaster using structured data and a machine learning model built in Python.

This project demonstrates an end-to-end ML pipeline including:
- Data preprocessing
- Exploratory data analysis (EDA)
- Model training using Random Forest
- Evaluation
- Model serialization using `joblib`

---

## Dataset Overview

- **Source**: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- **Records**: 891 passengers
- **Features**:  
  - `Pclass`: Ticket class  
  - `Sex`: Gender  
  - `Age`: Age in years  
  - `SibSp`: # of siblings/spouses aboard  
  - `Parch`: # of parents/children aboard  
  - `Fare`: Passenger fare  
  - `Embarked`: Port of embarkation  

- **Target Variable**: `Survived` (0 = No, 1 = Yes)

---

### Data Cleaning
- Filled missing `Age` values with median
- Filled missing `Embarked` values with mode
- Encoded `Sex` and `Embarked` using `LabelEncoder`

### Exploratory Data Analysis (EDA)
Visualized key patterns and relationships:
- Survival count distribution
- Survival by sex and passenger class
- Age distribution
- Feature correlation heatmap

### Model Building
- Used `RandomForestClassifier` from scikit-learn
- Trained on 80% of data, tested on 20%
- Saved final model as `titanic_model.pkl`

---

## Model Evaluation

- **Model**: Random Forest Classifier (`n_estimators=100`)
- **Train-Test Split**: 80/20
- **Accuracy**: **82.12%**

### Confusion Matrix

             Predicted
           |  0   |  1
     -----------------
    0 |   92  | 13  |
    1 |   19  | 55  |

### Classification Report
                precision    recall  f1-score   support

       0         0.83      0.88      0.85       105
       1         0.81      0.74      0.77        74

accuracy                             0.82       179
macro avg        0.82      0.81      0.81       179
weighted avg     0.82      0.82      0.82       179

---

## Charts Included
- `survival_count.png`
- `sex_vs_survival.png`
- `age_distribution.png`
- `correlation_matrix.png`

---
