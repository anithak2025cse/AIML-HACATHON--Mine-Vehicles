import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np

def draw_grid_background(ax, xlim=(0, 100), ylim=(0, 100), step=2.0):
    """Draws a subtle blueprint graph-paper grid pattern."""
    ax.set_facecolor('#F0F5FA')
    # Fine grid lines
    for x in np.arange(xlim[0], xlim[1] + step, step):
        ax.axvline(x, color='#E2ECF5', linewidth=0.7, zorder=0)
    for y in np.arange(ylim[0], ylim[1] + step, step):
        ax.axhline(y, color='#E2ECF5', linewidth=0.7, zorder=0)
    # Major grid lines every 10 units
    for x in np.arange(xlim[0], xlim[1] + 10, 10):
        ax.axvline(x, color='#D0E0EE', linewidth=1.2, zorder=0)
    for y in np.arange(ylim[0], ylim[1] + 10, 10):
        ax.axhline(y, color='#D0E0EE', linewidth=1.2, zorder=0)

def draw_card(ax, x, y, w, h, title, subtitle="", bg_color="#FFFFFF", border_color="#3B82F6", title_color="#1E3A8A", radius=0.6, lw=1.6):
    """Draws a stylized rounded card container with a title."""
    card = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                          edgecolor=border_color, facecolor=bg_color, linewidth=lw, zorder=2)
    ax.add_patch(card)
    if title:
        ax.text(x + w/2, y + h - 2.5, title, fontsize=9.5, weight='bold', ha='center', va='center', color=title_color, zorder=3)
    if subtitle:
        ax.text(x + w/2, y + h - 5.0, subtitle, fontsize=7.5, ha='center', va='center', color="#4B5563", zorder=3)
    return card

def create_diagram_2(output_path="reports/diagram2_mine_system_architecture.png"):
    """
    Creates Diagram 2: System Architecture blueprint flowchart matching the layout,
    colors, tech stack, and legend of Image 2.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    fig.patch.set_facecolor('#FFFFFF')
    draw_grid_background(ax)

    # -------------------------------------------------------------
    # TOP TITLE BANNER
    # -------------------------------------------------------------
    banner = FancyBboxPatch((4, 91.5), 92, 6.2, boxstyle="round,pad=0.3,rounding_size=0.8",
                           edgecolor="#0F172A", facecolor="#0F172A", linewidth=1.5, zorder=3)
    ax.add_patch(banner)
    ax.text(50, 94.6, "MINE VEHICLE FOG OPERATION & COLLISION SAFETY PIPELINE: SYSTEM ARCHITECTURE",
            fontsize=15.5, weight='black', ha='center', va='center', color='#FFFFFF', zorder=4)

    # -------------------------------------------------------------
    # RIGHT SIDE PANEL: LEGEND & TECH STACK
    # -------------------------------------------------------------
    # Legend Box
    leg_x, leg_y, leg_w, leg_h = 81.5, 71.0, 15.0, 17.5
    draw_card(ax, leg_x, leg_y, leg_w, leg_h, "Pipeline Legend", border_color="#64748B", bg_color="#FFFFFF", title_color="#0F172A", radius=0.6)
    
    # Legend Items
    items = [
        ("Data Flow", "#2563EB"),
        ("Model Flow", "#059669"),
        ("App Output", "#7C3AED")
    ]
    for idx, (label, col) in enumerate(items):
        iy = leg_y + leg_h - 6.5 - (idx * 3.8)
        # Arrow icon
        ax.annotate("", xy=(leg_x + 6.0, iy), xytext=(leg_x + 1.8, iy),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=12), zorder=4)
        ax.text(leg_x + 7.2, iy, label, fontsize=8.5, weight='bold', color="#1E293B", va='center', zorder=4)

    # Tech Stack Box
    tech_x, tech_y, tech_w, tech_h = 81.5, 12.0, 15.0, 56.0
    draw_card(ax, tech_x, tech_y, tech_w, tech_h, "Technology Stack", border_color="#64748B", bg_color="#FFFFFF", title_color="#0F172A", radius=0.6)
    
    techs = [
        ("Python 3.14", "Core Language & Engine", "#0284C7"),
        ("NumPy & Pandas", "Data Engine & Vector Physics", "#1E3A8A"),
        ("Scikit-Learn", "StandardScaler & Clustering", "#D97706"),
        ("TensorFlow / Keras", "Deep Artificial Neural Net", "#EA580C"),
        ("Flask REST Engine", "Backend API & Web Server", "#0F172A"),
        ("Plotly.js & CSS", "3D/2D Industrial Telemetry", "#7C3AED"),
        ("Digital-Twin Physics", "Friction & Braking Limits", "#059669")
    ]
    for tidx, (tname, tdesc, tcolor) in enumerate(techs):
        ty = tech_y + tech_h - 6.5 - (tidx * 7.2)
        # Badge pill
        t_pill = FancyBboxPatch((tech_x + 1.2, ty - 2.8), tech_w - 2.4, 5.8,
                               boxstyle="round,pad=0.1,rounding_size=0.4",
                               edgecolor=tcolor, facecolor=tcolor + "12", linewidth=1.2, zorder=3)
        ax.add_patch(t_pill)
        ax.text(tech_x + 2.0, ty + 1.0, f"● {tname}", fontsize=8.5, weight='bold', color=tcolor, va='center', zorder=4)
        ax.text(tech_x + 2.0, ty - 1.2, tdesc, fontsize=6.8, color="#4B5563", va='center', zorder=4)

    # -------------------------------------------------------------
    # ROW 1: DATA ENGINE & EXPLORATION
    # -------------------------------------------------------------
    # 1. Dataset Cylinder / Card
    draw_card(ax, 3.5, 73.5, 14.5, 15.0, "MINE TELEMETRY ENGINE", subtitle="(Synthetic Digital-Twin)",
              bg_color="#EFF6FF", border_color="#3B82F6", title_color="#1E40AF")
    ax.text(4.5, 81.0, "• 5,000 Multi-Sensor Logs\n• Fog Density: 0% to 100%\n• LiDAR LOS: 10m to 250m\n• Velocity: 0 to 60 km/h\n• Incline: -12° to +12°\n• Obstacle: 5m to 300m",
            fontsize=7.3, color="#1E293B", va='top', zorder=3)

    # 2. Data Validation Card
    draw_card(ax, 22.0, 73.5, 24.0, 15.0, "DATA VALIDATION & QUALITY CONTROL", subtitle="(Physical Boundary Auditing)",
              bg_color="#EFF6FF", border_color="#3B82F6", title_color="#1E40AF")
    ax.text(23.0, 81.0, "• Sensor dynamic range validation & null checks\n• Incline deceleration boundaries (a_eff > 0.4 m/s²)\n• Road friction bounds auditing (0.15 ≤ μ ≤ 0.70)\n• Safe stopping physics consistency checks",
            fontsize=7.3, color="#1E293B", va='top', zorder=3)

    # 3. EDA Card
    draw_card(ax, 50.0, 73.5, 27.0, 15.0, "EXPLORATORY DATA ANALYSIS (EDA)", subtitle="(Statistical & Meteorological Profiling)",
              bg_color="#EFF6FF", border_color="#3B82F6", title_color="#1E40AF")
    ax.text(51.0, 81.0, "• Fog vs. LiDAR attenuation decay curves\n• Incline grade vs. stopping distance distribution\n• Collision risk imbalance analysis (~8.7% positive class)\n• Multi-sensor correlation heatmaps",
            fontsize=7.3, color="#1E293B", va='top', zorder=3)

    # Arrows Row 1
    ax.annotate("", xy=(22.0, 81.0), xytext=(18.0, 81.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=4)
    ax.annotate("", xy=(50.0, 81.0), xytext=(46.0, 81.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=4)

    # Arrow from EDA down to Preprocessing
    ax.annotate("", xy=(63.5, 68.5), xytext=(63.5, 73.5),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=4)

    # -------------------------------------------------------------
    # ROW 2: PREPROCESSING PIPELINE
    # -------------------------------------------------------------
    prep_box = FancyBboxPatch((18.0, 56.5), 59.0, 12.0, boxstyle="round,pad=0.2,rounding_size=0.6",
                              edgecolor="#2563EB", facecolor="#F8FAFC", linewidth=1.5, zorder=2)
    ax.add_patch(prep_box)
    ax.text(47.5, 66.2, "DATA PREPROCESSING & FEATURE ENGINEERING PIPELINE",
            fontsize=9.5, weight='bold', ha='center', va='center', color="#1E40AF", zorder=3)

    # Sub-card 1: StandardScaler
    draw_card(ax, 19.5, 57.5, 26.5, 7.2, "FEATURE NORMALIZATION", subtitle="StandardScaler (Mean=0, Var=1)",
              bg_color="#EFF6FF", border_color="#93C5FD", title_color="#1E40AF", radius=0.4, lw=1.2)
    
    # Sub-card 2: Hazard Zoning
    draw_card(ax, 48.5, 57.5, 27.0, 7.2, "UNSUPERVISED HAZARD ZONING", subtitle="K-Means (k=3: Optimal, Caution, Critical)",
              bg_color="#EFF6FF", border_color="#93C5FD", title_color="#1E40AF", radius=0.4, lw=1.2)

    # Arrow inside preprocessing
    ax.annotate("", xy=(48.5, 61.1), xytext=(46.0, 61.1),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.8, mutation_scale=12), zorder=4)

    # Arrow from Preprocessing down to Train-Test Split
    ax.annotate("", xy=(32.5, 51.5), xytext=(32.5, 56.5),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=4)

    # -------------------------------------------------------------
    # ROW 3: TRAIN-TEST SPLIT & MODEL TRAINING
    # -------------------------------------------------------------
    # Train-Test Split Box
    draw_card(ax, 20.0, 42.5, 25.0, 9.0, "STRATIFIED TRAIN-TEST SPLIT", subtitle="(80/20 Preserving Risk Balance)",
              bg_color="#EFF6FF", border_color="#3B82F6", title_color="#1E40AF")
    ax.text(32.5, 44.5, "Training: 4,000 samples  |  Testing: 1,000 samples",
            fontsize=7.2, weight='semibold', ha='center', color="#1E293B", zorder=3)

    # Big Model Training Container
    mt_box = FancyBboxPatch((48.0, 31.0), 29.0, 20.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                            edgecolor="#D97706", facecolor="#FFFBEB", linewidth=1.5, zorder=2)
    ax.add_patch(mt_box)
    ax.text(62.5, 49.5, "MODEL TRAINING ENGINES", fontsize=9.5, weight='bold', ha='center', color="#B45309", zorder=3)

    # Model 1: Logistic Regression
    draw_card(ax, 49.5, 42.5, 12.5, 5.5, "Logistic Regression", subtitle="Linear Risk Baseline",
              bg_color="#FFFFFF", border_color="#F59E0B", title_color="#92400E", radius=0.3, lw=1.0)

    # Model 2: Linear Regression
    draw_card(ax, 63.0, 42.5, 12.5, 5.5, "Multiple Linear Reg", subtitle="Distance Predictor",
              bg_color="#FFFFFF", border_color="#F59E0B", title_color="#92400E", radius=0.3, lw=1.0)

    # Model 3: Keras Deep ANN
    draw_card(ax, 49.5, 32.5, 26.0, 8.5, "Keras Sequential Deep ANN", subtitle="64 ReLU → Drop 0.2 → 32 ReLU → 16 ReLU → 1 Sigmoid",
              bg_color="#FFFFFF", border_color="#D97706", title_color="#B45309", radius=0.4, lw=1.3)
    ax.text(62.5, 34.0, "Optimized with Adam + Binary Cross-Entropy (15 Epochs)",
            fontsize=7.0, ha='center', color="#4B5563", zorder=3)

    # Connecting arrow Train-Test to Model Training
    ax.annotate("", xy=(48.0, 42.0), xytext=(45.0, 45.0),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.2, mutation_scale=14), zorder=4)
    ax.text(45.5, 46.5, "Train Data (80%)", fontsize=7.5, weight='bold', color="#059669", zorder=4)

    # -------------------------------------------------------------
    # ROW 4: MODEL EVALUATION & SELECTION
    # -------------------------------------------------------------
    draw_card(ax, 3.5, 23.0, 38.0, 15.0, "MODEL EVALUATION & SELECTION DASHBOARD", subtitle="(Safety-Critical Validation)",
              bg_color="#ECFDF5", border_color="#10B981", title_color="#065F46")
    ax.text(4.5, 30.5, "• Primary Metric: RECALL (Minimizes Fatal False Negatives)\n• Baseline Comparison: Logistic Reg (Rec: ~99%) vs. Keras ANN (Rec: ~99%, Prec: ~73%)\n• Stopping Distance Metric: Linear Regression MAE = 2.80m, R² = 0.915\n• Keras Deep ANN selected to capture non-linear quadratic braking physics",
            fontsize=7.3, color="#064E3B", va='top', zorder=3)

    # Arrow from Train-Test Split down to Evaluation (Test Data)
    ax.annotate("", xy=(16.0, 38.0), xytext=(22.0, 42.5),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.2, mutation_scale=14), zorder=4)
    ax.text(14.0, 41.5, "Test Data (20%)", fontsize=7.5, weight='bold', color="#059669", zorder=4)

    # Arrow from Model Training (left border) to Model Evaluation (right border)
    ax.annotate("", xy=(41.5, 30.5), xytext=(48.0, 35.0),
                arrowprops=dict(arrowstyle="-|>", color="#059669", lw=2.2, mutation_scale=14), zorder=4)

    # -------------------------------------------------------------
    # ROW 5: PREDICTION PIPELINE, WEB APP, & INDUSTRIAL OUTPUTS
    # -------------------------------------------------------------
    # Prediction Pipeline REST API
    draw_card(ax, 3.5, 5.0, 20.0, 14.5, "PREDICTION PIPELINE", subtitle="(REST API / Flask Service)",
              bg_color="#EFF6FF", border_color="#2563EB", title_color="#1E40AF")
    ax.text(4.5, 12.8, "• In-memory model caching\n• Real-time REST endpoints:\n  /api/predict\n• Dynamic Safe Speed calculation:\n  v_safe = √(2 * a_eff * gap)\n• Instant micro-braking triggers",
            fontsize=7.1, color="#1E293B", va='top', zorder=3)

    # Arrow from Evaluation to Prediction Pipeline
    ax.annotate("", xy=(13.5, 19.5), xytext=(13.5, 23.0),
                arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=2.2, mutation_scale=14), zorder=4)

    # Web Application Dashboard
    draw_card(ax, 28.5, 5.0, 22.0, 14.5, "FLASK WEB APPLICATION", subtitle="(Industrial Dark-Themed UI)",
              bg_color="#FAF5FF", border_color="#7C3AED", title_color="#5B21B6")
    ax.text(29.5, 12.8, "• Page 1: Executive Summary & Twin\n• Page 2: 3D Unsupervised Zoning\n• Page 3: Model Benchmark Curves\n• Page 4: Live Telemetry Sandbox\n• Interactive Plotly visualizers",
            fontsize=7.1, color="#3B0764", va='top', zorder=3)

    # Arrow from Prediction Pipeline to Web App
    ax.annotate("", xy=(28.5, 12.2), xytext=(23.5, 12.2),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.2, mutation_scale=14), zorder=4)

    # 3 Output Cards
    out_x = 55.5
    out_w = 21.5

    # Output Card 1: Collision Risk
    draw_card(ax, out_x, 14.5, out_w, 5.0, "COLLISION RISK & ALERT LEVEL",
              bg_color="#FEF2F2", border_color="#EF4444", title_color="#991B1B", radius=0.3, lw=1.2)
    ax.text(out_x + out_w/2, 15.8, "CRITICAL (Flashing Red) / CAUTION / OPTIMAL",
            fontsize=7.2, weight='bold', color="#B91C1C", ha='center', zorder=4)

    # Output Card 2: Stopping Distance
    draw_card(ax, out_x, 8.8, out_w, 5.0, "SAFE STOPPING DISTANCE METRIC",
              bg_color="#F0FDF4", border_color="#10B981", title_color="#065F46", radius=0.3, lw=1.2)
    ax.text(out_x + out_w/2, 10.1, "Required Braking Dist vs. Radar Obstacle Gap",
            fontsize=7.2, weight='bold', color="#047857", ha='center', zorder=4)

    # Output Card 3: Dynamic Speed Advisory
    draw_card(ax, out_x, 3.2, out_w, 5.0, "DYNAMIC SPEED ADVISORY LIMIT",
              bg_color="#EFF6FF", border_color="#3B82F6", title_color="#1E40AF", radius=0.3, lw=1.2)
    ax.text(out_x + out_w/2, 4.5, "Safe Target Velocity to restore zero-risk margin",
            fontsize=7.2, weight='bold', color="#1D4ED8", ha='center', zorder=4)

    # Arrows from Web App to Output Cards
    ax.annotate("", xy=(out_x, 17.0), xytext=(50.5, 12.2),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.0, mutation_scale=12), zorder=4)
    ax.annotate("", xy=(out_x, 11.3), xytext=(50.5, 12.2),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.0, mutation_scale=12), zorder=4)
    ax.annotate("", xy=(out_x, 5.7), xytext=(50.5, 12.2),
                arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=2.0, mutation_scale=12), zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram 2 saved successfully to: {output_path}")

if __name__ == '__main__':
    create_diagram_2()
