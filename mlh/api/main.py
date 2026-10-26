import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

from api.schemas import (
    TelemetryInput,
    TelemetryBatchInput,
    PredictionOutput,
    ModelMetricsResponse,
    ClusterInfoResponse
)

app = FastAPI(
    title="Mine Vehicle Fog Operation & Collision Safety API",
    description="Physics-informed ML & Deep Learning API for haul truck safety in low-visibility mining environments.",
    version="1.0.0"
)

# Enable CORS for web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global artifacts container
ARTIFACTS = {}

def load_artifacts():
    global ARTIFACTS
    models_dir = 'models'
    
    scaler_cluster_path = os.path.join(models_dir, 'scaler_cluster.pkl')
    clusterer_path = os.path.join(models_dir, 'clusterer.pkl')
    scaler_model_path = os.path.join(models_dir, 'scaler_model.pkl')
    lin_reg_path = os.path.join(models_dir, 'linear_reg.pkl')
    log_reg_path = os.path.join(models_dir, 'logistic_reg.pkl')
    ann_mlp_path = os.path.join(models_dir, 'ann_mlp_model.pkl')
    ann_keras_path = os.path.join(models_dir, 'ann_keras_model.keras')
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    profiles_path = os.path.join(models_dir, 'cluster_profiles.json')

    if not os.path.exists(scaler_model_path):
        raise RuntimeError("Model artifacts not found. Please train models first using models/train_pipeline.py")

    ARTIFACTS['scaler_cluster'] = joblib.load(scaler_cluster_path)
    ARTIFACTS['clusterer'] = joblib.load(clusterer_path)
    ARTIFACTS['scaler_model'] = joblib.load(scaler_model_path)
    ARTIFACTS['linear_reg'] = joblib.load(lin_reg_path)
    ARTIFACTS['logistic_reg'] = joblib.load(log_reg_path)
    
    # Try loading Keras ANN model first, fallback to MLP
    if os.path.exists(ann_keras_path):
        try:
            import tensorflow as tf
            ARTIFACTS['ann_model'] = tf.keras.models.load_model(ann_keras_path)
            ARTIFACTS['ann_type'] = 'Keras Deep Neural Network'
        except Exception as e:
            print(f"Could not load Keras model: {e}")
            if os.path.exists(ann_mlp_path):
                ARTIFACTS['ann_model'] = joblib.load(ann_mlp_path)
                ARTIFACTS['ann_type'] = 'Scikit-Learn Deep MLP Neural Network'
    elif os.path.exists(ann_mlp_path):
        ARTIFACTS['ann_model'] = joblib.load(ann_mlp_path)
        ARTIFACTS['ann_type'] = 'Scikit-Learn Deep MLP Neural Network'
        
    with open(metrics_path, 'r') as f:
        ARTIFACTS['metrics'] = json.load(f)
        
    with open(profiles_path, 'r') as f:
        ARTIFACTS['cluster_profiles'] = json.load(f)

@app.on_event("startup")
def startup_event():
    load_artifacts()
    print("API startup completed: All ML & Neural Network models loaded successfully!")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Mine Vehicle Safety API",
        "ann_type": ARTIFACTS.get('ann_type', 'MLP Neural Network')
    }

def process_single_prediction(data: TelemetryInput) -> PredictionOutput:
    # 1. Scaled env features for Clustering
    env_arr = np.array([[data.Fog_Density, data.LiDAR_Range, data.Ramp_Incline]])
    env_scaled = ARTIFACTS['scaler_cluster'].transform(env_arr)
    cluster_id = int(ARTIFACTS['clusterer'].predict(env_scaled)[0]) if hasattr(ARTIFACTS['clusterer'], 'predict') else 0
    cluster_info = ARTIFACTS['cluster_profiles'].get(str(cluster_id), ARTIFACTS['cluster_profiles'].get(cluster_id, {}))
    cluster_name = cluster_info.get("name", f"Hazard Cluster {cluster_id}")
    
    # 2. Scaled full features for Supervised Regression & Classification
    feat_arr = np.array([[
        data.Fog_Density,
        data.LiDAR_Range,
        data.Truck_Velocity,
        data.Ramp_Incline,
        data.Obstacle_Distance
    ]])
    feat_scaled = ARTIFACTS['scaler_model'].transform(feat_arr)
    
    # Linear Regression for Safe Stopping Distance
    predicted_stopping_dist = float(ARTIFACTS['linear_reg'].predict(feat_scaled)[0])
    predicted_stopping_dist = max(predicted_stopping_dist, 2.0)
    
    # Neural Network for Collision Risk
    ann_model = ARTIFACTS['ann_model']
    if ARTIFACTS.get('ann_type') == 'Keras Deep Neural Network':
        risk_prob = float(ann_model.predict(feat_scaled, verbose=0)[0][0])
    else:
        risk_prob = float(ann_model.predict_proba(feat_scaled)[0][1])
        
    risk_prob_percent = round(risk_prob * 100.0, 2)
    threshold = 0.35  # Safety-first threshold
    risk_flag = 1 if risk_prob >= threshold or data.Obstacle_Distance < predicted_stopping_dist else 0
    
    # Dynamic Speed Recommendation Engine
    # Physics inverse: v_safe = sqrt(2 * a * max(0, Obstacle_Distance - margin))
    margin = 5.0
    usable_dist = max(1.0, data.Obstacle_Distance - margin)
    # Estimate effective deceleration based on incline and fog
    est_mu = 0.70 - 0.40 * (data.Fog_Density / 100.0) ** 1.5
    incline_rad = np.radians(data.Ramp_Incline)
    est_a = max(0.8, 9.81 * (est_mu * np.cos(incline_rad) + np.sin(incline_rad)))
    max_safe_v_ms = np.sqrt(2.0 * est_a * usable_dist)
    rec_speed_kmh = float(np.clip(max_safe_v_ms * 3.6, 0.0, 60.0))
    
    # Advisory Logic
    if risk_flag == 1 or risk_prob_percent > 60.0:
        advisory = f"CRITICAL HAZARD: Immediate deceleration required! Engage retarder/micro-brakes. Max safe speed: {rec_speed_kmh:.1f} km/h."
    elif risk_prob_percent > 30.0 or data.Fog_Density > 60.0:
        advisory = f"CAUTION: Restricted visibility or slope slip risk. Reduce velocity to {rec_speed_kmh:.1f} km/h and increase spacing."
    else:
        advisory = "NORMAL OPERATION: Line of sight clear. Maintain current speed with standard safety spacing."

    return PredictionOutput(
        safe_stopping_distance_meters=round(predicted_stopping_dist, 2),
        obstacle_distance_meters=round(data.Obstacle_Distance, 2),
        collision_risk_flag=risk_flag,
        collision_risk_probability_percent=risk_prob_percent,
        hazard_cluster_id=cluster_id,
        hazard_cluster_name=cluster_name,
        recommended_max_speed_kmh=round(rec_speed_kmh, 1),
        advisory_action=advisory,
        model_used=ARTIFACTS.get('ann_type', 'Deep Neural Network')
    )

@app.post("/predict", response_model=PredictionOutput)
def predict_hazard(data: TelemetryInput):
    try:
        return process_single_prediction(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionOutput])
def predict_hazard_batch(batch: TelemetryBatchInput):
    try:
        return [process_single_prediction(item) for item in batch.items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_metrics", response_model=ModelMetricsResponse)
def get_model_metrics():
    return ModelMetricsResponse(
        status="success",
        metrics=ARTIFACTS.get('metrics', {})
    )

@app.get("/hazard_clusters", response_model=ClusterInfoResponse)
def get_hazard_clusters():
    return ClusterInfoResponse(
        status="success",
        clusters=ARTIFACTS.get('cluster_profiles', {})
    )
