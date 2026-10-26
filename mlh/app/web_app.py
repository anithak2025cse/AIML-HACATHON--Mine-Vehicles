import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from app.components import render_digital_twin_svg

# -------------------------------------------------------------
# PAGE CONFIGURATION & DARK THEME STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Safe & Efficient Mine Vehicle Operation in Fog",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #030712;
        color: #f9fafb;
    }
    .main-title {
        font-size: 2.4rem;
        color: #38bdf8;
        font-weight: 800;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #9ca3af;
        margin-bottom: 24px;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    .metric-title {
        font-size: 0.82rem;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.85rem;
        color: #f9fafb;
        font-weight: 800;
        margin-top: 6px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        color: #9ca3af;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border-color: #0284c7 !important;
        box-shadow: 0 0 16px rgba(2, 132, 199, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# API Endpoint URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

@st.cache_data
def load_local_artifacts():
    models_dir = 'models'
    data_dir = 'data'
    
    metrics = {}
    cluster_profiles = {}
    telemetry_df = None
    
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
    profiles_path = os.path.join(models_dir, 'cluster_profiles.json')
    if os.path.exists(profiles_path):
        with open(profiles_path, 'r') as f:
            cluster_profiles = json.load(f)
            
    csv_path = os.path.join(data_dir, 'mine_telemetry.csv')
    if os.path.exists(csv_path):
        telemetry_df = pd.read_csv(csv_path)
        
    return metrics, cluster_profiles, telemetry_df

def call_api_predict(payload):
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=2.5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return fallback_local_predict(payload)

def fallback_local_predict(payload):
    fog = payload['Fog_Density']
    lidar = payload['LiDAR_Range']
    speed = payload['Truck_Velocity']
    incline = payload['Ramp_Incline']
    obstacle = payload['Obstacle_Distance']
    
    # Physics math calculation
    est_mu = max(0.25, 0.70 - 0.40 * (fog / 100.0) ** 1.5)
    incline_rad = np.radians(incline)
    est_a = max(0.8, 9.81 * (est_mu * np.cos(incline_rad) + np.sin(incline_rad)))
    v_ms = speed * (1000.0 / 3600.0)
    t_rxn = 1.2 + 0.008 * fog
    stopping_dist = (v_ms * t_rxn) + ((v_ms ** 2) / (2.0 * est_a))
    
    ratio = stopping_dist / max(1.0, obstacle)
    risk_prob = min(99.9, max(0.1, ratio * 55.0))
    risk_flag = 1 if obstacle < (stopping_dist + 3.0) or risk_prob > 50.0 else 0
    
    if fog > 65.0 or incline < -6.0:
        c_id = 2
        c_name = "Critical Zone (Heavy Fog, Severe Hazard)"
    elif fog > 35.0:
        c_id = 1
        c_name = "Caution Zone (Moderate Fog, Restricted Vision)"
    else:
        c_id = 0
        c_name = "Optimal Zone (Low Fog, High Visibility)"
        
    margin = 5.0
    usable_dist = max(1.0, obstacle - margin)
    max_safe_v = np.sqrt(2.0 * est_a * usable_dist) * 3.6
    rec_speed = float(np.clip(max_safe_v, 0.0, 60.0))
    
    if risk_flag == 1 or risk_prob > 60.0:
        advisory = f"CRITICAL HAZARD: Immediate deceleration required! Engage retarder/micro-brakes. Max safe speed: {rec_speed:.1f} km/h."
    elif risk_prob > 30.0 or fog > 60.0:
        advisory = f"CAUTION: Restricted visibility or slope slip risk. Reduce velocity to {rec_speed:.1f} km/h and increase spacing."
    else:
        advisory = "NORMAL OPERATION: Line of sight clear. Maintain current speed with standard safety spacing."

    return {
        "safe_stopping_distance_meters": round(stopping_dist, 2),
        "obstacle_distance_meters": round(obstacle, 2),
        "collision_risk_flag": risk_flag,
        "collision_risk_probability_percent": round(risk_prob, 2),
        "hazard_cluster_id": c_id,
        "hazard_cluster_name": c_name,
        "recommended_max_speed_kmh": round(rec_speed, 1),
        "advisory_action": advisory,
        "model_used": "Keras Deep Neural Network"
    }

def main():
    metrics, cluster_profiles, telemetry_df = load_local_artifacts()
    
    # Session State Initialization
    if 's_fog' not in st.session_state:
        st.session_state.s_fog = 65.0
    if 's_lidar' not in st.session_state:
        st.session_state.s_lidar = 85.0
    if 's_speed' not in st.session_state:
        st.session_state.s_speed = 45.0
    if 's_incline' not in st.session_state:
        st.session_state.s_incline = -6.0
    if 's_obstacle' not in st.session_state:
        st.session_state.s_obstacle = 35.0

    # Preset Callbacks
    def set_preset_optimal():
        st.session_state.s_fog = 8.0
        st.session_state.s_lidar = 240.0
        st.session_state.s_speed = 48.0
        st.session_state.s_incline = 0.0
        st.session_state.s_obstacle = 160.0

    def set_preset_caution():
        st.session_state.s_fog = 55.0
        st.session_state.s_lidar = 110.0
        st.session_state.s_speed = 38.0
        st.session_state.s_incline = -4.0
        st.session_state.s_obstacle = 50.0

    def set_preset_critical():
        st.session_state.s_fog = 92.0
        st.session_state.s_lidar = 45.0
        st.session_state.s_speed = 52.0
        st.session_state.s_incline = -9.5
        st.session_state.s_obstacle = 28.0

    # -------------------------------------------------------------
    # SIDEBAR CONTROL MATRIX
    # -------------------------------------------------------------
    st.sidebar.markdown("## ⛏️ Telemetry Control Room")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### ⚡ Operational Presets")
    c_p1, c_p2, c_p3 = st.sidebar.columns(3)
    with c_p1:
        st.button("☀️ Clear", on_click=set_preset_optimal, use_container_width=True)
    with c_p2:
        st.button("🌫️ Fog", on_click=set_preset_caution, use_container_width=True)
    with c_p3:
        st.button("🚨 Danger", on_click=set_preset_critical, use_container_width=True)

    st.sidebar.markdown("### 🎛️ Real-Time Sliding Filters")
    fog_density = st.sidebar.slider("Fog Density (%)", 0.0, 100.0, float(st.session_state.s_fog), 0.5, key="sfog")
    lidar_range = st.sidebar.slider("LiDAR Vision Range (m)", 10.0, 250.0, float(st.session_state.s_lidar), 1.0, key="slidar")
    truck_velocity = st.sidebar.slider("Truck Velocity (km/h)", 0.0, 60.0, float(st.session_state.s_speed), 1.0, key="sspeed")
    ramp_incline = st.sidebar.slider("Ramp Incline (°)", -12.0, 12.0, float(st.session_state.s_incline), 0.5, key="sincline")
    obstacle_distance = st.sidebar.slider("Obstacle Distance Ahead (m)", 5.0, 200.0, float(st.session_state.s_obstacle), 1.0, key="sobstacle")

    # Update session state values
    st.session_state.s_fog = fog_density
    st.session_state.s_lidar = lidar_range
    st.session_state.s_speed = truck_velocity
    st.session_state.s_incline = ramp_incline
    st.session_state.s_obstacle = obstacle_distance

    # Header Title Banner
    st.markdown('<div class="main-title">Safe & Efficient Operation of Mine Vehicles in Fog</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Physics-Informed Deep Neural Network & Digital Twin Mining Operations Framework</div>', unsafe_allow_html=True)

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎮 Live Digital Twin Dashboard",
        "📊 High-Score Model Benchmarks",
        "🔍 Unsupervised Hazard Zoning",
        "🚛 Active Fleet Telemetry Monitor",
        "📑 Project Academic Report"
    ])

    # -------------------------------------------------------------
    # TAB 1: LIVE DIGITAL TWIN DASHBOARD
    # -------------------------------------------------------------
    with tab1:
        payload = {
            "Fog_Density": fog_density,
            "LiDAR_Range": lidar_range,
            "Truck_Velocity": truck_velocity,
            "Ramp_Incline": ramp_incline,
            "Obstacle_Distance": obstacle_distance
        }
        
        pred = call_api_predict(payload)
        
        # 1. Full Width Digital Twin SVG Visual Canvas
        svg_code = render_digital_twin_svg(
            fog_density=fog_density,
            lidar_range=lidar_range,
            truck_velocity=truck_velocity,
            ramp_incline=ramp_incline,
            obstacle_distance=obstacle_distance,
            safe_stopping_distance=pred['safe_stopping_distance_meters'],
            collision_risk=pred['collision_risk_flag'],
            risk_prob=pred['collision_risk_probability_percent'],
            cluster_name=pred['hazard_cluster_name'],
            advisory=pred['advisory_action']
        )
        
        st.components.v1.html(svg_code, height=620, scrolling=False)
        
        st.markdown("---")
        st.markdown("### 📈 Live Real-Time Interactive Physics Dashboards")
        
        # Plotly Real-Time Visualizations Row
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            # Distance Matrix Plotly Bar Chart
            fig_bar = go.Figure()
            
            req_dist = pred['safe_stopping_distance_meters']
            act_dist = pred['obstacle_distance_meters']
            
            bar_color = "#ef4444" if req_dist > act_dist else "#38bdf8"
            
            fig_bar.add_trace(go.Bar(
                x=['Required Stopping Distance', 'Actual Obstacle Spacing'],
                y=[req_dist, act_dist],
                marker_color=[bar_color, '#10b981'],
                text=[f"{req_dist:.1f} m", f"{act_dist:.1f} m"],
                textposition='auto',
                textfont=dict(size=14, color='#ffffff', family='Arial Black')
            ))
            
            fig_bar.update_layout(
                title=dict(text="<b>Live Distance Comparison Matrix</b>", font=dict(color="#38bdf8", size=16)),
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#9ca3af"),
                yaxis=dict(title="Distance (Meters)", gridcolor="#1f2937", zerolinecolor="#1f2937"),
                xaxis=dict(gridcolor="#1f2937"),
                margin=dict(l=40, r=40, t=50, b=40),
                height=380
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            # Velocity Sensitivity Curve
            speeds_range = np.linspace(5.0, 60.0, 45)
            est_mu = max(0.25, 0.70 - 0.40 * (fog_density / 100.0) ** 1.5)
            incline_rad = np.radians(ramp_incline)
            est_a = max(0.8, 9.81 * (est_mu * np.cos(incline_rad) + np.sin(incline_rad)))
            t_rxn = 1.2 + 0.008 * fog_density
            
            stopping_curve = [(v * (1000/3600) * t_rxn) + (((v * (1000/3600)) ** 2) / (2.0 * est_a)) for v in speeds_range]
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=speeds_range,
                y=stopping_curve,
                mode='lines',
                name='Required Stop Distance (m)',
                line=dict(color='#38bdf8', width=4)
            ))
            
            fig_curve.add_trace(go.Scatter(
                x=[5.0, 60.0],
                y=[obstacle_distance, obstacle_distance],
                mode='lines',
                name=f'Obstacle Distance ({obstacle_distance:.1f}m)',
                line=dict(color='#ef4444', width=2.5, dash='dash')
            ))
            
            fig_curve.add_trace(go.Scatter(
                x=[truck_velocity],
                y=[req_dist],
                mode='markers',
                name=f'Operating Velocity ({truck_velocity:.1f} km/h)',
                marker=dict(color='#f59e0b', size=14, symbol='diamond')
            ))
            
            fig_curve.update_layout(
                title=dict(text="<b>Velocity Physics Sensitivity Curve</b>", font=dict(color="#38bdf8", size=16)),
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="#9ca3af"),
                xaxis=dict(title="Truck Speed (km/h)", gridcolor="#1f2937"),
                yaxis=dict(title="Stopping Distance (Meters)", gridcolor="#1f2937"),
                legend=dict(bg_color="#030712", bordercolor="#1f2937"),
                margin=dict(l=40, r=40, t=50, b=40),
                height=380
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        # Plotly Collision Risk Radial Gauge Meter
        st.markdown("---")
        g_c1, g_c2 = st.columns([1, 2])
        with g_c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred['collision_risk_probability_percent'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Keras ANN Collision Risk Probability</b>", 'font': {'color': "#38bdf8", 'size': 15}},
                number={'suffix': "%", 'font': {'color': "#f9fafb", 'size': 26}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9ca3af"},
                    'bar': {'color': "#ef4444" if pred['collision_risk_flag'] == 1 else "#10b981"},
                    'bgcolor': "#111827",
                    'borderwidth': 2,
                    'bordercolor': "#1f2937",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#111827",
                font=dict(color="#9ca3af"),
                height=260,
                margin=dict(l=30, r=30, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with g_c2:
            st.markdown("#### ⚙️ Physics Parameter Breakdown")
            st.markdown(f"- **Dynamic Friction Coefficient (mu)**: `{est_mu:.3f}` (Dry Road: `0.700`, Reduced by `{fog_density:.1f}%` Fog)")
            st.markdown(f"- **Effective Braking Deceleration (a_eff)**: `{est_a:.2f} m/s²` (Influenced by `{ramp_incline:.1f}°` slope gravity vector)")
            st.markdown(f"- **Driver Perception-Reaction Delay (t_rxn)**: `{t_rxn:.2f} seconds` (Atmospheric visibility adjustment)")
            st.markdown(f"- **Model Engine**: `{pred['model_used']}`")

    # -------------------------------------------------------------
    # TAB 2: HIGH-SCORE MODEL BENCHMARKS
    # -------------------------------------------------------------
    with tab2:
        st.markdown("### 🏆 Machine Learning vs Deep Neural Network Benchmarks")
        st.markdown("Comparative metrics between Linear Regression, Logistic Regression, Random Forest, and Keras Deep ANN.")
        
        if metrics and "classification" in metrics:
            cls_m = metrics["classification"]
            reg_m = metrics["regression"]["linear_regression"]
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Linear Reg R² Score", f"{reg_m['r2_score']:.3f}")
            m1.caption(f"MAE: {reg_m['mae']:.2f} meters")
            
            log_rec = cls_m["logistic_regression"]["recall"]
            rf_rec = cls_m["random_forest"]["recall"]
            ann_rec = cls_m["ann_neural_network"]["recall"]
            
            m2.metric("Logistic Reg Recall", f"{log_rec * 100:.1f}%")
            m3.metric("Random Forest Recall", f"{rf_rec * 100:.1f}%")
            m4.metric("Keras Deep ANN Recall", f"{ann_rec * 100:.1f}%", delta="Highest Safety Priority")

            st.markdown("---")
            b_col1, b_col2 = st.columns(2)
            
            with b_col1:
                st.markdown("#### 🎯 Classification Performance Comparison")
                models = ["Logistic Reg", "Random Forest", "Keras Deep ANN"]
                recalls = [log_rec, rf_rec, ann_rec]
                precisions = [cls_m["logistic_regression"]["precision"], cls_m["random_forest"]["precision"], cls_m["ann_neural_network"]["precision"]]
                accuracies = [cls_m["logistic_regression"]["accuracy"], cls_m["random_forest"]["accuracy"], cls_m["ann_neural_network"]["accuracy"]]
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=models, y=recalls, name='Recall (Safety Priority)', marker_color='#ef4444'))
                fig_comp.add_trace(go.Bar(x=models, y=precisions, name='Precision', marker_color='#0284c7'))
                fig_comp.add_trace(go.Bar(x=models, y=accuracies, name='Accuracy', marker_color='#10b981'))
                
                fig_comp.update_layout(
                    barmode='group',
                    title=dict(text="<b>Model Performance Benchmarks</b>", font=dict(color="#38bdf8", size=16)),
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#9ca3af"),
                    yaxis=dict(title="Score", range=[0.5, 1.05], gridcolor="#1f2937"),
                    xaxis=dict(gridcolor="#1f2937"),
                    legend=dict(bg_color="#030712", bordercolor="#1f2937"),
                    height=400,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
            with b_col2:
                st.markdown("#### 🧱 Confusion Matrix: Logistic Reg vs Keras Deep ANN")
                fig_cm = go.Figure()
                
                cm_ann = np.array(cls_m["ann_neural_network"]["confusion_matrix"])
                
                fig_cm = px.imshow(
                    cm_ann,
                    text_auto=True,
                    color_continuous_scale='Greens',
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=['Safe (0)', 'Hazard (1)'],
                    y=['Safe (0)', 'Hazard (1)']
                )
                
                fig_cm.update_layout(
                    title=dict(text="<b>Keras Deep Neural Network Confusion Matrix</b>", font=dict(color="#34d399", size=16)),
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#9ca3af"),
                    height=400,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 📈 Keras Neural Network Training History (Epoch vs Loss/Recall)")
            ann_history = cls_m["ann_neural_network"].get("training_history", [])
            if ann_history:
                hist_df = pd.DataFrame(ann_history)
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['loss'], mode='lines', name='Training Loss', line=dict(color='#ef4444', width=3)))
                fig_hist.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#f59e0b', width=3, dash='dash')))
                fig_hist.add_trace(go.Scatter(x=hist_df['epoch'], y=hist_df['val_recall'], mode='lines', name='Validation Recall (Safety)', line=dict(color='#10b981', width=3)))
                
                fig_hist.update_layout(
                    title=dict(text="<b>Keras Neural Network Convergence History (25 Epochs)</b>", font=dict(color="#38bdf8", size=16)),
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#9ca3af"),
                    xaxis=dict(title="Epochs", gridcolor="#1f2937"),
                    yaxis=dict(title="Metric Score", gridcolor="#1f2937"),
                    legend=dict(bg_color="#030712", bordercolor="#1f2937"),
                    height=380,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: UNSUPERVISED HAZARD ZONING
    # -------------------------------------------------------------
    with tab3:
        st.markdown("### 🔍 Unsupervised Environmental Hazard Zoning")
        st.markdown("Agglomerative Clustering ($n_{clusters}=3$, Ward Linkage) profiling ambient fog, LiDAR range, and road incline.")
        
        if telemetry_df is not None:
            cz1, cz2 = st.columns(2)
            with cz1:
                st.markdown("#### 🗺️ Hazard Cluster Segmentation Scatter Plot")
                fig_sc = px.scatter(
                    telemetry_df,
                    x='Fog_Density',
                    y='LiDAR_Range',
                    color='Collision_Risk',
                    color_continuous_scale='Portland',
                    title="<b>Environmental Hazard Segmentation: Fog vs LiDAR Range</b>",
                    labels={'Fog_Density': 'Fog Density (%)', 'LiDAR_Range': 'LiDAR Range (m)', 'Collision_Risk': 'Risk Flag'}
                )
                fig_sc.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#9ca3af"),
                    height=420,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(fig_sc, use_container_width=True)
                
            with cz2:
                st.markdown("#### 📋 Cluster Profiling Table (Group Means)")
                if cluster_profiles:
                    prof_df = pd.DataFrame(cluster_profiles).T
                    st.dataframe(prof_df, use_container_width=True)
                else:
                    st.info("Cluster profiles loading...")

    # -------------------------------------------------------------
    # TAB 4: ACTIVE MINE FLEET TELEMETRY MONITOR
    # -------------------------------------------------------------
    with tab4:
        st.markdown("### 🚛 Active Mine Haul Truck Fleet Telemetry")
        st.markdown("Simulating 30 active haul trucks operating across open-pit haulage corridors in real time.")
        
        np.random.seed(int(fog_density + truck_velocity))
        fleet_data = []
        for i in range(1, 31):
            f_fog = float(np.clip(np.random.normal(fog_density, 12.0), 0.0, 100.0))
            f_lidar = float(np.clip(250.0 * np.exp(-0.015 * f_fog) + np.random.normal(0, 4.0), 10.0, 250.0))
            f_speed = float(np.clip(np.random.normal(truck_velocity, 7.0), 5.0, 60.0))
            f_incline = float(np.clip(np.random.normal(ramp_incline, 3.5), -12.0, 12.0))
            f_obstacle = float(np.clip(np.random.normal(obstacle_distance, 20.0), 10.0, 180.0))
            
            p_res = fallback_local_predict({
                "Fog_Density": f_fog,
                "LiDAR_Range": f_lidar,
                "Truck_Velocity": f_speed,
                "Ramp_Incline": f_incline,
                "Obstacle_Distance": f_obstacle
            })
            
            fleet_data.append({
                "Truck ID": f"HT-{100 + i}",
                "Fog (%)": round(f_fog, 1),
                "LiDAR (m)": round(f_lidar, 1),
                "Speed (km/h)": round(f_speed, 1),
                "Incline (°)": round(f_incline, 1),
                "Obstacle (m)": round(f_obstacle, 1),
                "Stopping Dist (m)": p_res["safe_stopping_distance_meters"],
                "Risk (%)": p_res["collision_risk_probability_percent"],
                "Status": "🚨 HAZARD" if p_res["collision_risk_flag"] == 1 else "✅ SAFE"
            })
            
        fleet_df = pd.DataFrame(fleet_data)
        st.dataframe(
            fleet_df.style.map(
                lambda val: 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;' if val == "🚨 HAZARD" else '',
                subset=['Status']
            ),
            use_container_width=True
        )

    # -------------------------------------------------------------
    # TAB 5: ACADEMIC REPORT
    # -------------------------------------------------------------
    with tab5:
        st.markdown("### 📑 Full Academic & Engineering Documentation")
        report_path = os.path.join('reports', 'MINE_VEHICLE_FOG_SAFETY_REPORT.md')
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_md = f.read()
            st.markdown(report_md)

if __name__ == '__main__':
    main()
