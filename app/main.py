from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import time


# Create FastAPI application
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="Machine learning API for the Model Monitoring Lab Assignment",
    version="1.0.0"
)


# Load trained model
model_data = joblib.load("app/model.pkl")

model = model_data["model"]
feature_names = model_data["feature_names"]
target_names = model_data["target_names"]


# Request schema
class PredictionRequest(BaseModel):
    features: list[float]


# Home page
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">

    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>Breast Cancer Prediction API</title>

        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                color: #222;
            }

            .container {
                max-width: 900px;
                margin: 70px auto;
                padding: 20px;
            }

            .card {
                background: white;
                padding: 40px;
                border-radius: 14px;
                box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            }

            h1 {
                margin-top: 0;
                font-size: 32px;
            }

            .subtitle {
                color: #666;
                font-size: 17px;
                line-height: 1.6;
            }

            .status {
                display: inline-block;
                margin: 20px 0;
                padding: 8px 14px;
                border-radius: 20px;
                background: #e8f5e9;
                color: #2e7d32;
                font-weight: bold;
            }

            .section {
                margin-top: 30px;
            }

            .endpoint {
                background: #f7f7f7;
                padding: 14px 18px;
                margin: 10px 0;
                border-radius: 8px;
                font-family: monospace;
            }

            .method {
                font-weight: bold;
                margin-right: 12px;
            }

            .links {
                margin-top: 30px;
            }

            a {
                display: inline-block;
                text-decoration: none;
                padding: 12px 18px;
                margin-right: 10px;
                border-radius: 8px;
                background: #222;
                color: white;
            }

            a:hover {
                opacity: 0.85;
            }

            .footer {
                margin-top: 35px;
                color: #888;
                font-size: 14px;
            }
        </style>
    </head>

    <body>

        <div class="container">

            <div class="card">

                <h1>Breast Cancer Prediction API</h1>

                <p class="subtitle">
                    Machine learning API developed for the
                    Model Monitoring Lab Assignment.
                </p>

                <div class="status">
                    ● API is running
                </div>

                <div class="section">

                    <h2>About</h2>

                    <p>
                        This API serves a trained Logistic Regression
                        classification model using FastAPI.
                    </p>

                    <p>
                        The API is designed to demonstrate model serving,
                        latency monitoring and data drift detection.
                    </p>

                </div>

                <div class="section">

                    <h2>Available Endpoints</h2>

                    <div class="endpoint">
                        <span class="method">GET</span>
                        /
                    </div>

                    <div class="endpoint">
                        <span class="method">GET</span>
                        /health
                    </div>

                    <div class="endpoint">
                        <span class="method">POST</span>
                        /predict
                    </div>

                </div>

                <div class="links">

                    <a href="/docs">
                        API Documentation
                    </a>

                    <a href="/health">
                        Health Check
                    </a>

                </div>

                <div class="footer">
                    Model Monitoring Lab • FastAPI • Scikit-learn
                </div>

            </div>

        </div>

    </body>

    </html>
    """


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.perf_counter()

    # Validate number of features
    if len(request.features) != len(feature_names):
        return {
            "error": (
                f"Expected {len(feature_names)} features, "
                f"but received {len(request.features)}"
            )
        }

    # Convert input to NumPy array
    input_data = np.array(request.features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Calculate model latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "prediction": int(prediction),
        "prediction_label": target_names[prediction],
        "confidence": float(max(probabilities)),
        "model_latency_ms": round(latency_ms, 3)
    }