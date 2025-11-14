# Heart Disease Prediction API

This project is a midterm assignment for an end-to-end machine learning course. The goal is to build a complete ML-powered web service, from data analysis to a containerized API deployment, following MLOps best practices.

## 1\. Problem Description (Module 1)

Heart disease is a leading cause of death globally. Early detection can be life-saving. The objective of this project is to build a machine learning model that can predict the presence of heart disease in a patient based on a set of 13 clinical features.

This is a **binary classification** problem. The model will be trained to predict a `target` value of `1` (presence of disease) or `0` (absence of disease).

### Key Challenge: Evaluation (Module 4)

For this medical diagnosis task, **Recall** is the most important evaluation metric. A False Negative (failing to detect disease in a sick patient) is a much more critical and costly error than a False Positive. The final model is selected based on its ability to maximize Recall, while maintaining reasonable overall performance.

### The Dataset

The data is a consolidated heart disease dataset from Kaggle, which merges several databases including Cleveland, Hungary, and others.[1]

  * **Source:** [https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
  * **Data:** The dataset contains 13 input features and 1 target variable. It also contains missing values, which are automatically handled by a `SimpleImputer` within the model pipeline.[1]
  * **Features:**
      * **Numerical:** `age`, `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate), `oldpeak` (ST depression)
      * **Categorical:** `sex`, `cp` (chest pain type), `fbs` (fasting blood sugar \> 120), `restecg` (ECG results), `exang` (exercise angina), `slope`, `ca` (num. major vessels), `thal`
      * **Target:** `target` (0 = no disease, 1 = disease)

## 2\. How to Run the Project

This project uses `pipenv` for dependency management.

### Step 1: Install Dependencies

First, create your virtual environment and install all required packages from the `Pipfile.lock`.

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install project dependencies
pipenv install

```

### Step 2: Train the Model

This script will load the raw data from `data/`, run the full preprocessing and training pipeline (including imputation, scaling, and modeling), and save the final model as `model.pkl`.
```
# Activate your environment and run the training script
pipenv run python train.py
```