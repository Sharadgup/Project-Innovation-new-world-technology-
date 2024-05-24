from fastapi import FastAPI, HTTPException, Request
from pymongo import MongoClient 
import gridfs
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# MongoDB connection details
MONGO_URI = "mongodb+srv://shardgupta65:Typer%401345@cluster0.sp87qsr.mongodb.net/chatgpt"

# Retrieve the serialized model from MongoDB
def fetch_model_from_mongodb(model_name):
    try:
        client = MongoClient(MONGO_URI)
        db = client['chatgpt']
        fs = gridfs.GridFS(db)
        
        file_data = fs.find_one({'filename': model_name})

        if file_data is None:
            raise HTTPException(status_code=404, detail="Model file not found in MongoDB.")

        # Save the retrieved file locally
        model_path = 'retrieved_model.pkl'
        with open(model_path, 'wb') as f:
            f.write(file_data.read())

        print("Model retrieved from MongoDB successfully!")
        return model_path
    except Exception as e:
        print(f"Failed to retrieve the model from MongoDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve the model from MongoDB: {e}")

# Load the model using joblib
def load_model(model_name):
    model_path = fetch_model_from_mongodb(model_name)
    try:
        loaded_model = joblib.load(model_path)
        print("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load the model: {e}")

model_name = 'best_model.pkl'
model = load_model(model_name)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>House Price Prediction</title>
    </head>
    <body>
        <h1>House Price Prediction</h1>
        <form id="prediction-form">
            <label for="feature1">Feature 1:</label>
            <input type="text" id="feature1" name="feature1"><br><br>

            <label for="feature2">Feature 2:</label>
            <input type="text" id="feature2" name="feature2"><br><br>

            <!-- Add more input fields as necessary for your model features -->

            <button type="button" onclick="predict()">Predict</button>
        </form>

        <h2>Prediction Result:</h2>
        <p id="prediction-result"></p>

        <script>
            async function predict() {
                const form = document.getElementById('prediction-form');
                const formData = new FormData(form);
                const data = {};
                formData.forEach((value, key) => { data[key] = value });

                const response = await fetch('/predict/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                document.getElementById('prediction-result').innerText = result.prediction;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predict(request: Request):
    try:
        data = await request.json()
        input_data = pd.DataFrame(data, index=[0])

        # Preprocess input data
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)

        # Format response
        response_data = {"prediction": float(prediction[0])}

        # Return prediction as JSON response
        return response_data
    except Exception as e:
        print("Error during prediction:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
