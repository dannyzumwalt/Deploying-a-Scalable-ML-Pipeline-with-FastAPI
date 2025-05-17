# Environment Set up (pip or conda)
* Use the supplied file `environment.yml` to create a new environment with conda

# Deploying a Scalable ML Pipeline with FastAPI

This project demonstrates how to build, train, and deploy a machine learning pipeline using FastAPI. The model predicts whether an individual's income exceeds $50K/year based on demographic and employment data from the UCI Census Income dataset.

## Features

- Data preprocessing with one-hot encoding and label binarization
- Model training using a Random Forest Classifier
- RESTful API for inference using FastAPI
- Unit tests for core ML functions
- Example client script for local API interaction

## How to Run

1. **Install dependencies**  
   Make sure you have Python 3.10 and install requirements 

2. **Train the model**  
   ```
   python train_model.py
   ```

3. **Start the API**  
   ```
   uvicorn main:app --reload
   ```

4. **Test the API**  
   In a new terminal, run:
   ```
   python local_api.py
   ```

5. **Run unit tests**  
   ```
   pytest test_ml.py
   ```

## Notes

- The model and encoder files (`model.pkl`, `encoder.pkl`) are not tracked in git.
