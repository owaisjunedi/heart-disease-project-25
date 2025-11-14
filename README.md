# Heart Disease Prediction Service

A machine learning web service built with scikit-learn and FastAPI to predict a patient's risk of heart disease based on their clinical data.

## 📋 Description

Heart disease is a leading cause of death globally. Early detection is critical for improving patient outcomes, but it can be a complex diagnostic challenge. This project aims to solve this by building a reliable machine learning model to assist medical professionals in their assessment.

This service uses a **Random Forest Classifier** trained on the classic [UCI Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). The model analyzes 13 clinical features (such as age, cholesterol, chest pain type, and EKG results) to generate a binary prediction of whether a patient has heart disease.

The final, tuned model is packaged as a REST API using **FastAPI** and containerized with **Docker**, allowing it to be easily deployed as a microservice for any application to use.

## 🚀 Tech Stack

* **Data Analysis & Modeling:** Pandas, NumPy, scikit-learn, XGBoost
* **Web Service:** FastAPI, Uvicorn
* **Deployment:** Docker
* **Experimentation:** Jupyter Notebook
