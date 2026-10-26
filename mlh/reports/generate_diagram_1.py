import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, PathPatch
import matplotlib.path as mpath
import numpy as np

def draw_vector_icon(ax, icon_type, cx, cy, size=1.6):
    """Draws crisp vector monochrome icons inside boxes matching Image 1."""
    lw = 1.4
    if icon_type == "csv":
        # Document sheet with folded corner and CSV text
        w, h = size * 1.5, size * 2.0
        x, y = cx - w/2, cy - h/2
        doc = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                             edgecolor="#000000", facecolor="#FFFFFF", linewidth=lw, zorder=4)
        ax.add_patch(doc)
        # Folded line
        ax.plot([x + w - 0.6, x + w - 0.6, x + w], [y + h, y + h - 0.6, y + h - 0.6], color="#000000", lw=lw, zorder=5)
        # Inner text
        ax.text(cx, cy - 0.1, "CSV", fontsize=6.8, weight='black', ha='center', va='center', color="#000000", zorder=5)

    elif icon_type == "physics":
        # Gear / Incline slope symbol
        ax.plot([cx - size, cx + size], [cy - size*0.5, cy - size*0.5], color="#000000", lw=lw, zorder=4)
        ax.plot([cx - size, cx + size], [cy - size*0.5, cy + size*0.6], color="#000000", lw=lw, zorder=4)
        # Angle arc
        ax.plot([cx - size*0.3, cx - size*0.2], [cy - size*0.5, cy - size*0.2], color="#000000", lw=lw, zorder=4)
        # Vehicle box on incline
        car = FancyBboxPatch((cx - 0.2, cy - 0.1), size*0.8, size*0.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                             edgecolor="#000000", facecolor="#000000", lw=lw, zorder=5)
        ax.add_patch(car)

    elif icon_type == "funnel":
        # Filter funnel
        pts = np.array([
            [cx - size, cy + size*0.8],
            [cx + size, cy + size*0.8],
            [cx + size*0.25, cy - size*0.1],
            [cx + size*0.25, cy - size*0.8],
            [cx - size*0.25, cy - size*0.8],
            [cx - size*0.25, cy - size*0.1]
        ])
        funnel = Polygon(pts, closed=True, edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4)
        ax.add_patch(funnel)
        # Filter lines
        ax.plot([cx - size*0.6, cx + size*0.6], [cy + size*0.5, cy + size*0.5], color="#000000", lw=1.0, zorder=5)
        ax.plot([cx - size*0.4, cx + size*0.4], [cy + size*0.2, cy + size*0.2], color="#000000", lw=1.0, zorder=5)

    elif icon_type == "cluster":
        # 3 clustered nodes
        c1 = Circle((cx - size*0.6, cy - size*0.3), size*0.35, edgecolor="#000000", facecolor="#000000", lw=lw, zorder=4)
        c2 = Circle((cx + size*0.6, cy - size*0.3), size*0.35, edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4)
        c3 = Circle((cx, cy + size*0.6), size*0.35, edgecolor="#000000", facecolor="#000000", lw=lw, zorder=4)
        ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(c3)
        ax.plot([cx - size*0.6, cx], [cy - size*0.3, cy + size*0.6], color="#000000", lw=1.0, zorder=3)
        ax.plot([cx + size*0.6, cx], [cy - size*0.3, cy + size*0.6], color="#000000", lw=1.0, zorder=3)
        ax.plot([cx - size*0.6, cx + size*0.6], [cy - size*0.3, cy - size*0.3], color="#000000", lw=1.0, zorder=3)

    elif icon_type == "neural":
        # Sigmoid curve + Neural graph
        x_sig = np.linspace(cx - size, cx + size, 20)
        y_sig = cy + size * (1 / (1 + np.exp(-3*(x_sig - cx)/size)) - 0.5) * 1.5
        ax.plot(x_sig, y_sig, color="#000000", lw=lw*1.2, zorder=4)
        # Network nodes
        for ny in [cy - size*0.6, cy, cy + size*0.6]:
            ax.add_patch(Circle((cx - size, ny), 0.2, facecolor="#000000", zorder=5))
            ax.add_patch(Circle((cx + size, ny), 0.2, facecolor="#000000", zorder=5))

    elif icon_type == "eval":
        # Gear + Bar chart
        ax.add_patch(Circle((cx, cy), size*0.8, edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4))
        # Inner bars
        ax.plot([cx - 0.5, cx - 0.5], [cy - 0.5, cy + 0.1], color="#000000", lw=lw*1.6, zorder=5)
        ax.plot([cx, cx], [cy - 0.5, cy + 0.4], color="#000000", lw=lw*1.6, zorder=5)
        ax.plot([cx + 0.5, cx + 0.5], [cy - 0.5, cy + 0.6], color="#000000", lw=lw*1.6, zorder=5)

    elif icon_type == "trend":
        # Upward arrow trend graph with %
        ax.plot([cx - size, cx + size], [cy - size*0.6, cy - size*0.6], color="#000000", lw=lw, zorder=4)
        ax.plot([cx - size, cx - size], [cy - size*0.6, cy + size*0.6], color="#000000", lw=lw, zorder=4)
        ax.annotate("", xy=(cx + size*0.8, cy + size*0.6), xytext=(cx - size*0.7, cy - size*0.4),
                    arrowprops=dict(arrowstyle="-|>", color="#000000", lw=lw*1.3, mutation_scale=10), zorder=5)
        ax.text(cx + 0.2, cy - 0.1, "%", fontsize=8.5, weight='black', color="#000000", zorder=5)

    elif icon_type == "dashboard":
        # Monitor display with alert exclamation
        w, h = size * 1.8, size * 1.3
        x, y = cx - w/2, cy - h/2 + 0.2
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                                    edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4))
        # Base stand
        ax.plot([cx - size*0.4, cx + size*0.4], [cy - h/2, cy - h/2], color="#000000", lw=lw*1.3, zorder=4)
        ax.plot([cx, cx], [cy - h/2 + 0.2, cy - h/2], color="#000000", lw=lw*1.3, zorder=4)
        # Warning triangle inside
        tri = Polygon([[cx, cy + 0.4], [cx - 0.45, cy - 0.3], [cx + 0.45, cy - 0.3]], closed=True,
                      edgecolor="#000000", facecolor="#000000", lw=0.8, zorder=5)
        ax.add_patch(tri)
        ax.text(cx, cy - 0.1, "!", fontsize=6.5, weight='black', color="#FFFFFF", ha='center', va='center', zorder=6)

    elif icon_type == "truck":
        # Heavy mining haul truck icon
        ax.add_patch(FancyBboxPatch((cx - 1.2, cy - 0.4), 1.6, 0.9, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    edgecolor="#000000", facecolor="#000000", lw=lw, zorder=4))
        ax.add_patch(FancyBboxPatch((cx + 0.4, cy - 0.4), 0.8, 0.7, boxstyle="round,pad=0.05,rounding_size=0.1",
                                    edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4))
        # Wheels
        ax.add_patch(Circle((cx - 0.7, cy - 0.6), 0.3, facecolor="#000000", zorder=5))
        ax.add_patch(Circle((cx + 0.7, cy - 0.6), 0.3, facecolor="#000000", zorder=5))

    elif icon_type == "bulb":
        # Lightbulb explainability icon
        ax.add_patch(Circle((cx, cy + 0.2), size*0.5, edgecolor="#000000", facecolor="#FFFFFF", lw=lw, zorder=4))
        ax.plot([cx - 0.3, cx + 0.3], [cy - 0.3, cy - 0.3], color="#000000", lw=lw, zorder=5)
        ax.plot([cx - 0.2, cx + 0.2], [cy - 0.5, cy - 0.5], color="#000000", lw=lw, zorder=5)
        # Ray lines
        ax.plot([cx, cx], [cy + 0.8, cy + 1.1], color="#000000", lw=lw, zorder=4)
        ax.plot([cx - 0.7, cx - 1.0], [cy + 0.6, cy + 0.8], color="#000000", lw=lw, zorder=4)
        ax.plot([cx + 0.7, cx + 1.0], [cy + 0.6, cy + 0.8], color="#000000", lw=lw, zorder=4)

def create_diagram_1(output_path="reports/diagram1_mine_vehicle_pipeline.png"):
    """Creates crisp monochrome pipeline diagram matching Image 1."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(20, 11.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    # Main Title
    ax.text(50, 96, "AI-BASED MINE VEHICLE COLLISION RISK PREDICTION AND SAFETY ASSESSMENT SYSTEM",
            fontsize=18, weight='bold', ha='center', va='center', family='sans-serif', color='#000000')

    # -------------------------------------------------------------
    # 8 SEQUENTIAL TOP BOXES
    # -------------------------------------------------------------
    box_w = 9.8
    box_h = 42.0
    start_x = 2.2
    gap_x = 2.5
    y_top = 48.0

    stages = [
        {
            "num": "1. Telemetry Dataset",
            "icon_type": "csv",
            "points": [
                "• 5,000+ mine records",
                "• 5 sensor attributes",
                "• Fog: 0% to 100%",
                "• LiDAR: 10m to 250m",
                "• Velocity: 0 to 60 km/h",
                "• Incline: -12° to +12°",
                "• Obstacle: 5m to 300m"
            ]
        },
        {
            "num": "2. Physics Engine",
            "icon_type": "physics",
            "points": [
                "• Friction decay model:",
                "  μ = 0.6*(1 - Fog/150)",
                "• Dynamic deceleration:",
                "  a = g*(μ*cosθ + sinθ)",
                "• Downhill slip penalty",
                "• Reaction delay time",
                "• Safe stop distance"
            ]
        },
        {
            "num": "3. Preprocessing",
            "icon_type": "funnel",
            "points": [
                "• Sensor boundary check",
                "• Missing/NaN validation",
                "• StandardScaler fit",
                "• Stratified split (80:20)",
                "• Preserving risk balance",
                "• Outlier suppression"
            ]
        },
        {
            "num": "4. Hazard Zoning",
            "icon_type": "cluster",
            "points": [
                "• Environmental features",
                "• K-Means / Agglomerative",
                "• n_clusters = 3",
                "• Cluster 0: Optimal",
                "• Cluster 1: Caution",
                "• Cluster 2: Critical",
                "• Unsupervised profiling"
            ]
        },
        {
            "num": "5. Model Training",
            "icon_type": "neural",
            "points": [
                "• Multiple Linear Reg",
                "  (Predict Stop Dist)",
                "• Logistic Regression",
                "  (Baseline Risk)",
                "• Keras Deep ANN:",
                "  64 ReLU → Drop 0.2",
                "  → 32 ReLU → 16 ReLU",
                "  → 1 Sigmoid output"
            ]
        },
        {
            "num": "6. Evaluation & Selection",
            "icon_type": "eval",
            "points": [
                "• RECALL (Top Priority!)",
                "  Avoid False Negatives",
                "• Precision & F1-Score",
                "• ROC-AUC & Confusion",
                "• R² & MAE for Regression",
                "• Keras ANN selected for",
                "  non-linear physics"
            ]
        },
        {
            "num": "7. Risk Prediction",
            "icon_type": "trend",
            "points": [
                "• predict_proba() output",
                "• Collision risk score",
                "• Non-linear threshold",
                "• Dynamic Safe Speed:",
                "  v_safe = √(2*a*gap)",
                "• Micro-braking trigger"
            ]
        },
        {
            "num": "8. Safety Output",
            "icon_type": "dashboard",
            "points": [
                "• Critical Collision Risk",
                "  → MICRO-BRAKING",
                "• Restricted Visibility",
                "  → SPEED REDUCTION",
                "• Optimal Line of Sight",
                "  → NORMAL CRUISE"
            ]
        }
    ]

    for i, stg in enumerate(stages):
        bx = start_x + i * (box_w + gap_x)
        # Rectangle
        rect = FancyBboxPatch((bx, y_top), box_w, box_h,
                              boxstyle="round,pad=0.2,rounding_size=0.6",
                              edgecolor="#000000", facecolor="#FFFFFF", linewidth=1.8, zorder=2)
        ax.add_patch(rect)
        
        # Header Box Line
        ax.plot([bx, bx + box_w], [y_top + box_h - 4.5, y_top + box_h - 4.5], color='#000000', linewidth=1.2, zorder=3)
        
        # Stage Title
        ax.text(bx + box_w/2, y_top + box_h - 2.5, stg["num"],
                fontsize=9.2, weight='bold', ha='center', va='center', color='#000000', zorder=4)
        
        # Vector Icon
        draw_vector_icon(ax, stg["icon_type"], bx + box_w/2, y_top + box_h - 8.0, size=1.4)
        
        # Bullet Points
        text_y = y_top + box_h - 12.5
        for pt in stg["points"]:
            ax.text(bx + 0.6, text_y, pt, fontsize=7.6, ha='left', va='top', color='#111111', family='sans-serif', zorder=4)
            text_y -= 4.0

        # Connecting Arrow to next box
        if i < len(stages) - 1:
            arrow_start_x = bx + box_w
            arrow_end_x = bx + box_w + gap_x
            ax.annotate("", xy=(arrow_end_x, y_top + box_h/2), xytext=(arrow_start_x, y_top + box_h/2),
                        arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.8, mutation_scale=14), zorder=5)

    # -------------------------------------------------------------
    # BOTTOM CONTAINER 1: CHALLENGE-SPECIFIC PROCESSING (DASHED)
    # -------------------------------------------------------------
    dashed_x = 2.2
    dashed_y = 5.0
    dashed_w = 69.5
    dashed_h = 36.5
    
    dashed_rect = FancyBboxPatch((dashed_x, dashed_y), dashed_w, dashed_h,
                                 boxstyle="round,pad=0.2,rounding_size=0.8",
                                 edgecolor="#000000", facecolor="#FFFFFF", linewidth=1.6, linestyle='--', zorder=2)
    ax.add_patch(dashed_rect)
    
    # Title of dashed container
    ax.text(dashed_x + dashed_w/2, dashed_y + dashed_h - 2.8,
            "Challenge-Specific Processing (Real-Time Telemetry & Safety Explainability)",
            fontsize=12, weight='bold', ha='center', va='center', color='#000000', zorder=3)

    # Sub-boxes inside dashed container
    sub_w = 13.5
    sub_h = 26.5
    sub_y = dashed_y + 3.4
    
    sub_boxes = [
        {
            "x": dashed_x + 2.5,
            "icon_type": "truck",
            "title": "Mine Telemetry Inputs",
            "desc": [
                "• Current Fog Density (%)",
                "• LiDAR Detection (m)",
                "• Truck Velocity (km/h)",
                "• Ramp Grade Angle (°)",
                "• Obstacle Radar Spacing"
            ]
        },
        {
            "x": dashed_x + 19.5,
            "icon_type": "neural",
            "title": "Trained Keras Model",
            "desc": [
                "• Evaluates dynamic road μ",
                "• Computes friction decay",
                "• Captures non-linear v²",
                "• Deceleration gradient",
                "• Deep ANN forward pass"
            ]
        },
        {
            "x": dashed_x + 36.5,
            "icon_type": "trend",
            "title": "Class Probabilities",
            "desc": [
                "• Collision Risk Probability",
                "• predict_proba() confidence",
                "• Safety threshold τ = 0.35",
                "• Required Stopping Dist",
                "• Safe Velocity Target"
            ]
        },
        {
            "x": dashed_x + 53.5,
            "icon_type": "bulb",
            "title": "WHY IS THIS RISK?",
            "desc": [
                "• Stopping Dist > Obstacle",
                "• Slick moisture friction drop",
                "• Steep downhill gravity slip",
                "• Non-linear velocity penalty",
                "• Micro-braking advisory"
            ]
        }
    ]

    for j, sb in enumerate(sub_boxes):
        s_rect = FancyBboxPatch((sb["x"], sub_y), sub_w, sub_h,
                                boxstyle="round,pad=0.2,rounding_size=0.5",
                                edgecolor="#000000", facecolor="#FFFFFF", linewidth=1.4, zorder=3)
        ax.add_patch(s_rect)
        
        # Sub-box header icon & title
        draw_vector_icon(ax, sb["icon_type"], sb["x"] + sub_w/2, sub_y + sub_h - 3.2, size=1.2)
        ax.text(sb["x"] + sub_w/2, sub_y + sub_h - 6.6, sb["title"], fontsize=9.2, weight='bold', ha='center', va='center', zorder=4)
        
        # Line
        ax.plot([sb["x"], sb["x"] + sub_w], [sub_y + sub_h - 8.6, sub_y + sub_h - 8.6], color='#000000', linewidth=1.0, zorder=3)
        
        # Desc
        dy = sub_y + sub_h - 11.2
        for line in sb["desc"]:
            ax.text(sb["x"] + 0.6, dy, line, fontsize=7.5, ha='left', va='top', color='#222222', zorder=4)
            dy -= 2.75

        # Arrow to next sub-box
        if j < len(sub_boxes) - 1:
            ax.annotate("", xy=(sb["x"] + sub_w + 3.5, sub_y + sub_h/2), xytext=(sb["x"] + sub_w, sub_y + sub_h/2),
                        arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.6, mutation_scale=12), zorder=5)

    # Bottom caption inside dashed box
    ax.text(dashed_x + dashed_w/2, dashed_y + 2.0,
            "Physics-informed dynamic inference pipeline with real-time speed advisory & micro-braking thresholds",
            fontsize=8.8, style='italic', ha='center', va='center', color='#333333', zorder=4)

    # -------------------------------------------------------------
    # BOTTOM CONTAINER 2: WEB APPLICATION DASHBOARD
    # -------------------------------------------------------------
    app_x = 74.5
    app_y = 5.0
    app_w = 23.3
    app_h = 36.5

    app_rect = FancyBboxPatch((app_x, app_y), app_w, app_h,
                              boxstyle="round,pad=0.2,rounding_size=0.8",
                              edgecolor="#000000", facecolor="#FFFFFF", linewidth=1.8, zorder=2)
    ax.add_patch(app_rect)

    # App Header
    ax.text(app_x + app_w/2, app_y + app_h - 3.0, "Mine Safety Web Dashboard", fontsize=11.5, weight='bold', ha='center', va='center', zorder=3)
    ax.text(app_x + app_w/2, app_y + app_h - 5.5, "(Flask & Modern Web Stack)", fontsize=8.5, style='italic', ha='center', va='center', color='#444444', zorder=3)
    ax.plot([app_x, app_x + app_w], [app_y + app_h - 7.5, app_y + app_h - 7.5], color='#000000', linewidth=1.2, zorder=3)

    # 4 Module Badges
    modules = [
        {"title": "1. Executive Summary", "sub": "Physics data twin & dataset metrics"},
        {"title": "2. Hazard Zoning", "sub": "3D Unsupervised cluster visualization"},
        {"title": "3. Model Benchmarking", "sub": "Recall metrics & Keras loss curves"},
        {"title": "4. Telemetry Sandbox", "sub": "Real-time haul truck simulation"}
    ]

    mod_y = app_y + app_h - 13.0
    for mod in modules:
        m_rect = FancyBboxPatch((app_x + 1.5, mod_y), app_w - 3.0, 4.6,
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                edgecolor="#000000", facecolor="#FAFAFA", linewidth=1.2, zorder=3)
        ax.add_patch(m_rect)
        ax.text(app_x + 2.8, mod_y + 2.8, f"●  {mod['title']}", fontsize=8.8, weight='bold', ha='left', va='center', zorder=4)
        ax.text(app_x + 2.8, mod_y + 1.1, mod['sub'], fontsize=7.2, color='#555555', ha='left', va='center', zorder=4)
        mod_y -= 5.8

    # Connecting arrows from Step 8 to Web App and from dashed box to Step 4 & 7
    ax.annotate("", xy=(app_x + app_w/2, app_y + app_h), xytext=(app_x + app_w/2, y_top),
                arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.8, mutation_scale=14), zorder=5)

    ax.annotate("", xy=(dashed_x + dashed_w/2, y_top), xytext=(dashed_x + dashed_w/2, dashed_y + dashed_h),
                arrowprops=dict(arrowstyle="<|-|>", color="#000000", lw=1.6, mutation_scale=14, linestyle='--'), zorder=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagram 1 saved successfully to: {output_path}")

if __name__ == '__main__':
    create_diagram_1()
