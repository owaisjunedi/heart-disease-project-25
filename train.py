import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

print("Starting training script...")

# --- 1. Load and Clean Data ---
# Load data
df = pd.read_csv('data/heart_disease_uci.csv')

# Define critical features (13 predictors)
# MAKE SURE 'thalch' or your corrected column name is here
critical_features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
    'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal' 
]

# Clean the data
df_cleaned = df.dropna(subset=critical_features)
df_cleaned = df_cleaned.copy()

# Target engineering (Remove num column and set it as target and refine it as per requirement)
# Per the UCI spec [1], 0 is absence, 1-4 is presence.
df_cleaned['target'] = df_cleaned['num'].apply(lambda x: 1 if x > 0 else 0)

# Create our final DataFrame
df_final = df_cleaned[critical_features + ['target']]
df = df_final

# --- 2. Split the Data ---
# We use the full 80% (train+val) for training the final model
X = df.drop('target', axis=1)
y = df['target']

# Split into 80% training (X_train_full) and 20% test --> Combining train & validation dataset for final
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_full = X_train_full.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train_full = y_train_full.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# --- 3. Process Data (Fit Preprocessors) ---
numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# 1. Handle Categorical Data
# Convert data to list of dictionaries
train_dicts = X_train_full[categorical_features].to_dict(orient='records')

# Initialize and fit DictVectorizer
dv = DictVectorizer(sparse=False)
X_train_cat = dv.fit_transform(train_dicts)

# 2. Handle Numerical Data
# Initialize and fit StandardScaler
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_full[numerical_features])

# 3. Combine the processed data
X_train_final = np.hstack((X_train_num, X_train_cat))

# --- 4. Train Final Model ---

# These are the best parameters we found from GridSearchCV
best_params = {
    'n_estimators': 60,         # Number of trees in the forest
    'max_depth': 5,             # Maximum depth of each tree
    'min_samples_leaf': 4       # Min samples required at a leaf node
}

# Initialize the final model with the best parameters
final_model = RandomForestClassifier(random_state=42, **best_params)

# Train on the full 80% training set
final_model.fit(X_train_final, y_train_full)
print("Model training complete.")

# --- 5. Save (Serialize) the Model and Preprocessors ---

# We need to save 3 files:
# 1. The DictVectorizer (dv)
# 2. The StandardScaler (scaler)
# 3. The Model (final_model)

with open('dv.pkl', 'wb') as f_out:
    pickle.dump(dv, f_out)
print("Saved DictVectorizer to dv.pkl")

with open('scaler.pkl', 'wb') as f_out:
    pickle.dump(scaler, f_out)
print("Saved StandardScaler to scaler.pkl")

with open('model.pkl', 'wb') as f_out:
    pickle.dump(final_model, f_out)
print("Saved final model to model.pkl")

print("\nTraining complete. All artifacts saved.")