import pandas as pd
import numpy as np
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

# --- 1. Load Artifacts ---
print("Loading model artifacts...")
# Load the DictVectorizer (Categorical Features - One Hot Encoding)
with open('dv.pkl', 'rb') as f_in:
    dv = pickle.load(f_in)

# Load the StandardScaler (For Numerical Features)
with open('scaler.pkl', 'rb') as f_in:
    scaler = pickle.load(f_in)

# Load the Model (Our FINAL trained model)
with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)
print("Artifacts loaded successfully.")

# --- 2. Define Input Data Model ---
# This defines the exact structure and data types
# for a request to our API.
# It MUST match the 13 features your model was trained on.
class PatientData(BaseModel):
    age: int
    sex: str  # e.g., 'Male', 'Female'
    cp: str   # e.g., 'Typical Angina', 'Atypical Angina'
    trestbps: int
    chol: int
    fbs: str  # e.g., 'True', 'False'
    restecg: str # e.g., 'Normal', 'ST-T Abnormality'
    thalch: int
    exang: str # e.g., 'Yes', 'No'
    oldpeak: float
    slope: str # e.g., 'Upsloping', 'Flat'
    ca: float # This was a float in the original
    thal: str # e.g., 'Normal', 'Fixed Defect'

# --- 3. Create FastAPI App ---
app = FastAPI(title="Heart Disease Prediction API", version="1.0")


# Define a simple root endpoint
# We commented this below because we created a new webpage for this, if 4.1 wasn't done, we would have kept the below def read_root()
# @app.get("/")
# def read_root():
#     """
#     Root endpoint with a welcome message.
#     """
#     return {"message": "Heart Disease Prediction API is running. Go to /docs for details."}

# ---  4.1 Define the ROOT endpoint (to serve the HTML) ---
# This is done for calling this API from the STATIC index.html website we created.
@app.get("/", include_in_schema=False)  # include_in_schema=False hides it from API docs
async def get_frontend():
    """
    Serves the main static index.html webpage.
    """
    return FileResponse('index.html')

# --- 4. Define Prediction Endpoint ---
@app.post("/predict")
def predict_heart_disease(data: PatientData):
    """
    Makes a prediction on a single patient's data.
    The input data is validated by the PatientData model.
    """

    # 1. Convert Pydantic data to a dictionary
    patient_dict = data.dict()
    
    # 2. Separate numerical and categorical features
    # These lists MUST match your training script
    numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Create dictionaries for processing
    cat_data_dict = {k: patient_dict[k] for k in categorical_features}
    num_data_df = pd.DataFrame([patient_dict], columns=numerical_features)
    
    # 3. Process the data
    # Transform categorical data with DictVectorizer
    # Note: dv expects a list of dictionaries
    X_cat = dv.transform([cat_data_dict])
    
    # Transform numerical data with StandardScaler
    X_num = scaler.transform(num_data_df)
    
    # 4. Combine processed data
    X_final = np.hstack((X_num, X_cat))
    
    # 5. Make prediction
    # We use predict_proba to get the probability
    probability = model.predict_proba(X_final)[0, 1] # Get prob of class '1'
    prediction = int(probability >= 0.5) # Get a binary 0/1 prediction
    
    # 6. Return response
    return {
        "heart_disease_probability": float(probability),
        "heart_disease_prediction": prediction
    }

# --- 5. Run the App with uvicorn---
# This allows the script to be run directly with 'python predict.py'
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)