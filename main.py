from fastapi import FastAPI, HTTPException, Request
from pymongo import MongoClient
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load MongoDB model path
def fetch_model_path_from_mongodb(model_name):
    try:
        client = MongoClient("mongodb+srv://shardgupta65:Typer%401345@cluster0.sp87qsr.mongodb.net/chatgpt")
        db = client['chatgpt']
        collection = db['fs.files']
        
        model_data = collection.find_one({'filename': model_name})
        
        if model_data:
            model_path = model_data.get('metadata', {}).get('model_path')
            if model_path:
                print("Model path fetched successfully:", model_path)
                return model_path
            else:
                print("Model path not found in the database entry")
                raise HTTPException(status_code=404, detail="Model path not found in the database entry")
        else:
            print("Model with the specified name not found")
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        print("Error while fetching model path from MongoDB:", str(e))
        raise HTTPException(status_code=500, detail="Error fetching model path from MongoDB")

def load_model(model_name):
    model_path = fetch_model_path_from_mongodb(model_name)
    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Model loaded successfully from:", model_path)
        return model
    else:
        print("Model path does not exist")
        raise HTTPException(status_code=404, detail="Model path does not exist")

model_name = 'retrieved_model.pkl'
model = load_model(model_name)

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
