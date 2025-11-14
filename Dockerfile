# 1. Use an official lightweight Python image
# We use a 'slim' version for a smaller image size
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the file that lists our dependencies
# This layer is cached and only re-runs if requirements.txt changes
COPY requirements.txt .

# 4. Install the dependencies from the requirements file
# This is much faster than pipenv
# We add --no-cache-dir to keep the image size small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the training script and data
# This layer only re-runs if 'train.py' or the data changes.
COPY train.py /app/
COPY data/heart_disease_uci.csv /app/data/heart_disease_uci.csv

# 6. Train the model *inside* the container
# This bakes the model.pkl files directly into the image.
RUN python train.py

# 7. Copy the *rest* of your application code
# This layer is very fast. It only re-runs if predict.py changes.
# It will NOT trigger the 'RUN python train.py' step above.
COPY predict.py /app/

# 7.1 Copy the Webpage we created for our API
COPY index.html /app/

# 8. Expose port 8080 (the port our FastAPI app runs on -> from predict.py)
EXPOSE 8080

# 9. The command to run when the container starts
# This runs our 'predict.py' file using uvicorn
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]