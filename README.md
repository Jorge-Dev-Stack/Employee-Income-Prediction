# Employee Income Prediction (Machine Learning Project)

This project demonstrates a complete **machine learning pipeline** built with Python and scikit-learn.  
It predicts **employee income** based on various demographic and job-related features, handling missing data and categorical encoding automatically.

---

## Project Overview

The goal of this project is to:

- Load and explore an employee dataset with missing values  
- Clean and preprocess the data using `pandas` and `scikit-learn`  
- Train a **Random Forest Regressor** model to predict employee income  
- Evaluate model performance using **Mean Absolute Error (MAE)**  
- Save the trained model for future deployment with `joblib`

This is an excellent beginner-friendly example of an **end-to-end supervised learning workflow**.

---

## Features Used

- **pandas** for data loading and cleaning  
- **scikit-learn** for preprocessing, pipeline creation, and model training  
- **RandomForestRegressor** for regression modeling  
- **joblib** for model serialization (saving the trained model)

---

## Machine Learning Workflow

1. **Load the dataset** from a public GitHub CSV file  
2. **Clean the data**, ensuring the target variable (`income`) has no missing values  
3. **Split the data** into training and testing subsets  
4. **Preprocess the features**:
   - Numerical features: impute missing values using the **median**
   - Categorical features: impute missing values using the **most frequent value** and apply **One-Hot Encoding**
5. **Train a Random Forest Regressor**
6. **Evaluate the model** using Mean Absolute Error (MAE)
7. **Save the trained model** for future use

---

## Technologies Used

| Library | Purpose |
|----------|----------|
| `pandas` | Data loading and cleaning |
| `scikit-learn` | Machine learning and preprocessing |
| `joblib` | Saving and loading trained models |
| `Python 3.10+` | Programming language |

---

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/employee-income-prediction.git
   cd employee-income-prediction

