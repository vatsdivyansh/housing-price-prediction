from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib 

import pandas as pd
import numpy as np
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="Predict house prices using ML model",
    version="1.0.0"
)

# Add CORS middleware (for Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model = joblib.load('housing_price_model.pkl')
    print(f"✓ Model loaded from: housing_price_model.pkl")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Define request schema
class HousePredictionRequest(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: Optional[float] = None  # Can be missing (will be imputed)
    population: float
    households: float
    median_income: float
    ocean_proximity: str

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.230,
                "latitude": 37.880,
                "housing_median_age": 41,
                "total_rooms": 880,
                "total_bedrooms": 129,
                "population": 322,
                "households": 126,
                "median_income": 8.3252,
                "ocean_proximity": "NEAR BAY"
            }
        }

# Define response schema
class PredictionResponse(BaseModel):
    predicted_price: float
    price_in_lakhs: float
    message: str

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_house_price(request: HousePredictionRequest):
    """
    Predict housing price based on input features.
    
    Parameters:
    - longitude: X coordinate (East-West)
    - latitude: Y coordinate (North-South)
    - housing_median_age: Age of housing in years
    - total_rooms: Total number of rooms
    - total_bedrooms: Total number of bedrooms (optional, can be None)
    - population: Number of residents
    - households: Number of households
    - median_income: Median income (in tens of thousands)
    - ocean_proximity: Proximity to ocean (categorical)
    
    Returns:
    - predicted_price: Price in original units (100s of dollars)
    - price_in_lakhs: Price converted to Indian Rupees (lakhs)
    - message: Formatted prediction message
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Create DataFrame from request
        input_data = pd.DataFrame([{
            "longitude": request.longitude,
            "latitude": request.latitude,
            "housing_median_age": request.housing_median_age,
            "total_rooms": request.total_rooms,
            "total_bedrooms": request.total_bedrooms,
            "population": request.population,
            "households": request.households,
            "median_income": request.median_income,
            "ocean_proximity": request.ocean_proximity
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Convert to lakhs (original data is in 100s of dollars)
        prediction_lakhs = prediction / 100000
        
        return PredictionResponse(
            predicted_price=float(prediction),
            price_in_lakhs=float(prediction_lakhs),
            message=f"Predicted price: ${prediction:,.2f} (₹{prediction_lakhs:.2f} L)"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict-batch")
def predict_batch(requests: list[HousePredictionRequest]):
    """
    Predict for multiple houses at once.
    
    Takes a list of HousePredictionRequest objects.
    Returns predictions for all houses.
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        input_data = pd.DataFrame([
            {
                "longitude": r.longitude,
                "latitude": r.latitude,
                # "housing_median_age": r.housing_median_age,
                "total_rooms": r.total_rooms,
                "total_bedrooms": r.total_bedrooms,
                "population": r.population,
                "households": r.households,
                "median_income": r.median_income,
                "ocean_proximity": r.ocean_proximity
            }
            for r in requests
        ])
        
        predictions = model.predict(input_data)
        
        return {
            "predictions": [float(p) for p in predictions],
            "count": len(predictions),
            "message": f"Predicted {len(predictions)} houses successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(e)}"
        )

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "docs": "/docs (Interactive API documentation)"
        },
        "example_feature_values": {
            "longitude": -122.230,
            "latitude": 37.880,
            "housing_median_age": 41,
            "total_rooms": 880,
            "total_bedrooms": 129,
            "population": 322,
            "households": 126,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
