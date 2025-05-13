# Housing Price Prediction API

This project involves creating a machine learning-based API using **Flask** that estimates housing prices based on given features. The model is trained with housing data, and the API returns predictions along with confidence scores.

## Main Goal

The goal of this project is to allow users to submit **POST** requests with a set of input features and receive **predictions** (housing prices) and the associated **confidence** levels for those predictions.

## Model Details

A **Random Forest Regressor** is employed for this project, which predicts housing prices using 13 input features.

- The trained model is stored as `model2.pkl` and is loaded when the Flask application is started.

## Setup Instructions

### Step 1: Train the Model
Run `train-2.py`. This will save the trained model as `model2.pkl` in the `app` directory.

### Step 2: Launch the Flask App
In your terminal, navigate to the project directory and execute `app.py`:

### Step 3: Build Docker image
docker build -t ml-model .

### Step 4: Run Docker container
docker run -p 9000:9000 ml-model

## Sample API Request and Response

### **Prediction Endpoint:**

- **Method**: `POST`
- **URL**: `http://127.0.0.1:9000/predict`

### **Request Body Example**

```json
{
  "features": [
    [8500, 5, 3, 2, 1, 1, 1, 1, 1, 2, 0, 1, 0]
  ]
}

### **Response Example**
```json
{
    "confidence": 0.0,
    "prediction": 9682680.0
}

confidence = confidence score 0 = low and 1 = high
prediction = predicted house price

##Health Check
To make sure API is running
http://localhost:9000/health
Response
{
    "status": "ok"
}

ok mean its running

Tech
pip install -r requirements.txt

Jialin Bai 64070503409
