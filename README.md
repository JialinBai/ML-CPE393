# Housing Price Prediction API

This project involves creating a machine learning-based API using **Flask** that estimates housing prices based on given features. The model is trained with housing data, and the API returns predictions along with confidence scores.

## Main Goal

The goal of this project is to allow users to submit **POST** requests with a set of input features and receive **predictions** (housing prices) and the associated **confidence** levels for those predictions.

## Model Details

A **Random Forest Regressor** is employed for this project, which predicts housing prices using 13 input features.

- The trained model is stored as `model2.pkl` and is loaded when the Flask application is started.

## Setup Instructions

### Step 1: Train the Model
Run `train.py`. This will save the trained model as `model2.pkl` in the `app` directory.

### Step 2: Launch the Flask App
In your terminal, navigate to the project directory and execute `app.py`:

```bash
cd "project folder directory"
python app.py
