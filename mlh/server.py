import os
import io
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Keras TensorFlow Import
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global In-Memory Caching Store for Production-Grade Performance
CACHE = {}

def generate_physics_dataset(num_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a physics-sound multi-sensor dataset mimicking an open-pit mine digital twin.
    """
    np.random.seed(random_seed)
    
    # 1. Fog Density (Continuous percentage: 0% to 100%)
    fog_density = np.random.uniform(0.0, 100.0, num_samples)
    
    # 2. LiDAR Range (10m to 250m, inversely proportional to Fog_Density with random noise)
    lidar_attenuation = np.exp(-0.015 * fog_density)
    lidar_noise = np.random.normal(0, 5.0, num_samples)
    lidar_range = np.clip(250.0 * lidar_attenuation + lidar_noise, 10.0, 250.0)
    
    # 3. Truck Velocity (0 to 60 km/h)
    truck_velocity = np.random.uniform(5.0, 60.0, num_samples)
    
    # 4. Ramp Incline (-12° to +12°)
    ramp_incline = np.random.uniform(-12.0, 12.0, num_samples)
    
    # 5. Obstacle Distance (5m to 300m)
    obstacle_distance = np.random.uniform(5.0, 300.0, num_samples)
    
    # --- PHYSICS CALCULATIONS ---
    # Friction Coefficient mu = 0.6 * (1 - (Fog_Density / 150))
    mu = 0.6 * (1.0 - (fog_density / 150.0)) + np.random.normal(0, 0.02, num_samples)
    mu = np.clip(mu, 0.15, 0.70)
    
    v_ms = truck_velocity / 3.6
    g = 9.81
    
    # Downhill grade penalty (if Ramp_Incline < 0, add incline gravity factor)
    incline_rad = np.radians(ramp_incline)
    effective_decel = g * (mu * np.cos(incline_rad) + np.sin(incline_rad))
    effective_decel = np.clip(effective_decel, 0.4, 8.5)
    
    # Safe Stopping Distance = (v_ms**2) / (2 * 9.81 * mu) + (reaction distance)
    reaction_dist = v_ms * (1.2 + 0.006 * fog_density)
    braking_dist = (v_ms ** 2) / (2.0 * effective_decel)
    stopping_noise = np.random.normal(0, 1.0, num_samples)
    safe_stopping_distance = np.clip(reaction_dist + braking_dist + stopping_noise, 2.0, 320.0)
    
    # Binary Target: Collision_Risk = 1 if Obstacle_Distance < Safe_Stopping_Distance else 0
    collision_risk = (obstacle_distance < (safe_stopping_distance + 3.0)).astype(int)
    
    df = pd.DataFrame({
        'Fog_Density': np.round(fog_density, 2),
        'LiDAR_Range': np.round(lidar_range, 2),
        'Truck_Velocity': np.round(truck_velocity, 2),
        'Ramp_Incline': np.round(ramp_incline, 2),
        'Obstacle_Distance': np.round(obstacle_distance, 2),
        'Friction_Mu': np.round(mu, 3),
        'Safe_Stopping_Distance': np.round(safe_stopping_distance, 2),
        'Collision_Risk': collision_risk
    })
    
    return df

def initialize_models_and_data():
    """
    Initializes dataset, unsupervised hazard clustering, regression, and Keras ANN models.
    """
    if 'dataset' in CACHE:
        return
        
    print("Generating 5,000-row physics dataset...")
    df = generate_physics_dataset(5000)
    CACHE['dataset'] = df

    # --- PHASE 2: UNSUPERVISED HAZARD CLUSTERING ---
    env_cols = ['Fog_Density', 'LiDAR_Range', 'Ramp_Incline']
    scaler_cluster = StandardScaler()
    X_env_scaled = scaler_cluster.fit_transform(df[env_cols])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_env_scaled)
    df['Hazard_Cluster'] = cluster_labels
    
    # Map cluster numbers to domain profiles based on Fog_Density mean
    profiles = df.groupby('Hazard_Cluster')[env_cols + ['Friction_Mu', 'Safe_Stopping_Distance', 'Collision_Risk']].mean()
    sorted_clusters = sorted(profiles.index, key=lambda k: profiles.loc[k, 'Fog_Density'])
    
    cluster_mapping = {
        sorted_clusters[0]: {"name": "Optimal Zone", "desc": "Low fog, high visibility, flat ground", "code": 0, "color": "#10b981"},
        sorted_clusters[1]: {"name": "Caution Zone", "desc": "Moderate fog, restricted sight", "code": 1, "color": "#f59e0b"},
        sorted_clusters[2]: {"name": "Critical Zone", "desc": "Heavy blinding fog, steep downhill slopes", "code": 2, "color": "#ef4444"}
    }
    
    CACHE['scaler_cluster'] = scaler_cluster
    CACHE['clusterer'] = kmeans
    CACHE['cluster_mapping'] = cluster_mapping
    CACHE['cluster_profiles'] = profiles.to_dict()

    # --- PHASE 3: SUPERVISED ML & DEEP LEARNING (80/20 SPLIT) ---
    feature_cols = ['Fog_Density', 'LiDAR_Range', 'Truck_Velocity', 'Ramp_Incline', 'Obstacle_Distance']
    X = df[feature_cols]
    y_reg = df['Safe_Stopping_Distance']
    y_cls = df['Collision_Risk']
    
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    scaler_model = StandardScaler()
    X_train_scaled = scaler_model.fit_transform(X_train)
    X_test_scaled = scaler_model.transform(X_test)
    
    CACHE['scaler_model'] = scaler_model
    CACHE['X_test_scaled'] = X_test_scaled
    CACHE['y_reg_test'] = y_reg_test
    CACHE['y_cls_test'] = y_cls_test
    
    # 1. Multiple Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_reg_train)
    y_reg_pred = lin_reg.predict(X_test_scaled)
    
    lin_metrics = {
        "mae": float(mean_absolute_error(y_reg_test, y_reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
        "r2": float(r2_score(y_reg_test, y_reg_pred))
    }
    CACHE['lin_reg'] = lin_reg
    CACHE['lin_metrics'] = lin_metrics
    
    # 2. Logistic Regression
    log_reg = LogisticRegression(class_weight='balanced', random_state=42)
    log_reg.fit(X_train_scaled, y_cls_train)
    y_log_pred = log_reg.predict(X_test_scaled)
    
    log_metrics = {
        "accuracy": float(accuracy_score(y_cls_test, y_log_pred)),
        "precision": float(precision_score(y_cls_test, y_log_pred)),
        "recall": float(recall_score(y_cls_test, y_log_pred)),
        "f1": float(f1_score(y_cls_test, y_log_pred)),
        "confusion_matrix": confusion_matrix(y_cls_test, y_log_pred).tolist()
    }
    CACHE['log_reg'] = log_reg
    CACHE['log_metrics'] = log_metrics
    
    # 3. Keras/TensorFlow Sequential ANN
    # Architecture: 64 ReLU -> 0.2 Dropout -> 32 ReLU -> 16 ReLU -> 1 Sigmoid output
    ann = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')])
    
    print("Training Keras Sequential ANN for 15 epochs...")
    history = ann.fit(
        X_train_scaled, y_cls_train,
        epochs=15,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    ann_prob = ann.predict(X_test_scaled, verbose=0).flatten()
    ann_pred = (ann_prob >= 0.35).astype(int)
    
    ann_metrics = {
        "accuracy": float(accuracy_score(y_cls_test, ann_pred)),
        "precision": float(precision_score(y_cls_test, ann_pred)),
        "recall": float(recall_score(y_cls_test, ann_pred)),
        "f1": float(f1_score(y_cls_test, ann_pred)),
        "confusion_matrix": confusion_matrix(y_cls_test, ann_pred).tolist(),
        "training_history": [
            {
                "epoch": ep + 1,
                "loss": float(history.history['loss'][ep]),
                "val_loss": float(history.history['val_loss'][ep]),
                "recall": float(history.history['recall'][ep]),
                "val_recall": float(history.history['val_recall'][ep])
            }
            for ep in range(15)
        ]
    }
    
    CACHE['ann'] = ann
    CACHE['ann_metrics'] = ann_metrics
    print("Initialization Complete! All models trained and cached in memory.")

# -------------------------------------------------------------
# FLASK ROUTE HANDLERS
# -------------------------------------------------------------
@app.before_request
def ensure_initialized():
    initialize_models_and_data()

@app.route('/')
def page_executive_summary():
    """Page 1: 📊 Executive Summary & Data Twin"""
    df = CACHE['dataset']
    total_records = len(df)
    avg_fog = round(float(df['Fog_Density'].mean()), 2)
    overall_risk_pct = round(float(df['Collision_Risk'].mean() * 100.0), 2)
    
    # Sample 500 rows for high-performance Plotly scatter plot rendering
    sample_df = df.sample(n=600, random_state=42).copy()
    scatter_data = sample_df.to_dict(orient='records')
    
    return render_template(
        'page1_summary.html',
        total_records=total_records,
        avg_fog=avg_fog,
        overall_risk_pct=overall_risk_pct,
        scatter_data=json.dumps(scatter_data),
        active_page='summary'
    )

@app.route('/hazard-zoning')
def page_hazard_zoning():
    """Page 2: 🔍 Hazard Zoning (Unsupervised Clustering)"""
    df = CACHE['dataset']
    sample_df = df.sample(n=600, random_state=42).copy()
    scatter_3d_data = sample_df[['Fog_Density', 'LiDAR_Range', 'Ramp_Incline', 'Hazard_Cluster', 'Collision_Risk']].to_dict(orient='records')
    
    cluster_mapping = CACHE['cluster_mapping']
    cluster_profiles = df.groupby('Hazard_Cluster')[['Fog_Density', 'LiDAR_Range', 'Ramp_Incline', 'Friction_Mu', 'Safe_Stopping_Distance', 'Collision_Risk']].mean().round(2).to_dict(orient='index')
    
    return render_template(
        'page2_clustering.html',
        scatter_3d_data=json.dumps(scatter_3d_data),
        cluster_mapping=cluster_mapping,
        cluster_profiles=cluster_profiles,
        active_page='clustering'
    )

@app.route('/benchmarking')
def page_benchmarking():
    """Page 3: 🧠 Model Benchmarking & DL Training"""
    lin_metrics = CACHE['lin_metrics']
    log_metrics = CACHE['log_metrics']
    ann_metrics = CACHE['ann_metrics']
    
    return render_template(
        'page3_benchmark.html',
        lin_metrics=lin_metrics,
        log_metrics=log_metrics,
        ann_metrics=ann_metrics,
        history_json=json.dumps(ann_metrics['training_history']),
        active_page='benchmarking'
    )

@app.route('/live-telemetry')
def page_live_telemetry():
    """Page 4: 🎛️ Live Dispatch Telemetry (Sandbox)"""
    return render_template(
        'page4_sandbox.html',
        active_page='telemetry'
    )

@app.route('/architecture')
def page_architecture():
    """Page 5: 📐 System Architecture & Pipeline Flowcharts"""
    return render_template(
        'page5_architecture.html',
        active_page='architecture'
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Live REST API Prediction Endpoint"""
    data = request.json
    fog = float(data.get('Fog_Density', 65.0))
    lidar = float(data.get('LiDAR_Range', 85.0))
    velocity = float(data.get('Truck_Velocity', 45.0))
    incline = float(data.get('Ramp_Incline', -6.0))
    obstacle = float(data.get('Obstacle_Distance', 35.0))
    
    # 1. Physics Calculations
    mu = 0.6 * (1.0 - (fog / 150.0))
    mu = max(0.15, min(0.70, mu))
    v_ms = velocity / 3.6
    g = 9.81
    incline_rad = np.radians(incline)
    effective_decel = max(0.4, g * (mu * np.cos(incline_rad) + np.sin(incline_rad)))
    
    safe_stopping_distance = round(float((v_ms ** 2) / (2.0 * effective_decel) + v_ms * (1.2 + 0.006 * fog)), 2)
    
    # 2. ANN Prediction
    feat_scaled = CACHE['scaler_model'].transform([[fog, lidar, velocity, incline, obstacle]])
    ann_prob = float(CACHE['ann'].predict(feat_scaled, verbose=0)[0][0])
    ann_prob_pct = round(ann_prob * 100.0, 1)
    
    collision_risk = 1 if (obstacle < safe_stopping_distance or ann_prob >= 0.35) else 0
    
    # 3. Recommended Target Velocity to bring risk back to 0
    margin = 5.0
    usable_dist = max(1.0, obstacle - margin)
    max_safe_v_ms = np.sqrt(max(0.0, 2.0 * effective_decel * usable_dist))
    recommended_velocity = round(min(60.0, float(max_safe_v_ms * 3.6)), 1)
    
    if collision_risk == 1:
        alert_title = "COLLISION IMMINENT - ACTIVATE MICRO-BRAKING"
        alert_status = "CRITICAL"
        alert_class = "danger"
    else:
        alert_title = "OPERATIONAL EFFICIENCY OPTIMAL"
        alert_status = "SAFE"
        alert_class = "success"

    return jsonify({
        "safe_stopping_distance": safe_stopping_distance,
        "obstacle_distance": obstacle,
        "collision_risk": collision_risk,
        "risk_probability_percent": ann_prob_pct,
        "recommended_velocity": recommended_velocity,
        "alert_title": alert_title,
        "alert_status": alert_status,
        "alert_class": alert_class,
        "effective_friction": round(mu, 3)
    })

@app.route('/download-csv')
def download_csv():
    """Generates CSV dataset download on demand"""
    df = CACHE['dataset']
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='mine_vehicle_fog_telemetry.csv'
    )

if __name__ == '__main__':
    initialize_models_and_data()
    app.run(host='127.0.0.1', port=5000, debug=False)
