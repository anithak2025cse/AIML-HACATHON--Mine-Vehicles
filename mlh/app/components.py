import numpy as np

def render_digital_twin_svg(
    fog_density: float,
    lidar_range: float,
    truck_velocity: float,
    ramp_incline: float,
    obstacle_distance: float,
    safe_stopping_distance: float,
    collision_risk: int,
    risk_prob: float,
    cluster_name: str,
    advisory: str
) -> str:
    """
    Renders a high-definition, responsive SVG Digital Twin Telemetry Canvas.
    """
    # Color palette based on hazard classification
    if collision_risk == 1:
        alert_bg = "rgba(127, 29, 29, 0.95)"
        alert_border = "#ef4444"
        alert_text = "#fca5a5"
        badge_title = "🚨 CRITICAL COLLISION HAZARD"
    elif risk_prob > 30.0 or fog_density > 60.0:
        alert_bg = "rgba(120, 53, 15, 0.95)"
        alert_border = "#f59e0b"
        alert_text = "#fde68a"
        badge_title = "⚠️ RESTRICTED VISIBILITY / SLIP CAUTION"
    else:
        alert_bg = "rgba(6, 78, 59, 0.95)"
        alert_border = "#10b981"
        alert_text = "#a7f3d0"
        badge_title = "✅ OPTIMAL OPERATIONAL ZONE"

    fog_opacity = round(float(np.clip(fog_density / 100.0 * 0.82, 0.05, 0.85)), 2)
    
    # Scale physical coordinates (Canvas width = 900px, height = 280px)
    max_dist = max(250.0, obstacle_distance + 35.0)
    scale_factor = 720.0 / max_dist
    
    truck_x = 70.0
    stopping_width = max(20.0, safe_stopping_distance * scale_factor)
    obstacle_x = min(770.0, truck_x + (obstacle_distance * scale_factor))
    lidar_width = min(750.0, lidar_range * scale_factor)

    # Ramp incline angle rotation (-12° to +12°)
    incline_angle = -float(np.clip(ramp_incline * 1.5, -12.0, 12.0))

    # Calculate recommended safe speed
    margin = 5.0
    usable_dist = max(1.0, obstacle_distance - margin)
    ratio = usable_dist / max(1.0, safe_stopping_distance)
    rec_speed = float(np.clip(truck_velocity * np.sqrt(ratio), 0.0, 60.0))

    svg_html = f"""
    <div style="background-color: #0b0f19; border-radius: 14px; padding: 24px; border: 1px solid #1e293b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f8fafc; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);">
        
        <!-- Header Banner -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 12px;">
            <div>
                <h3 style="margin: 0; color: #38bdf8; font-size: 1.4rem; font-weight: 800; letter-spacing: -0.3px;">📡 Digital Twin Vehicle Sensor Matrix</h3>
                <span style="font-size: 0.9rem; color: #94a3b8;">Real-Time Atmospheric & Physics Constraint Engine</span>
            </div>
            <div style="background-color: {alert_bg}; border: 2px solid {alert_border}; color: {alert_text}; padding: 10px 20px; border-radius: 30px; font-weight: 800; font-size: 0.95rem; letter-spacing: 0.5px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
                {badge_title} ({risk_prob:.1f}% Risk)
            </div>
        </div>

        <!-- SVG Digital Twin Canvas -->
        <div style="position: relative; width: 100%; background: linear-gradient(180deg, #111827 0%, #030712 100%); border-radius: 12px; overflow: hidden; border: 1px solid #1f2937; box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);">
            <svg viewBox="0 0 900 270" style="width: 100%; height: auto; display: block;">
                <!-- Grid Distance Ruler -->
                <line x1="0" y1="210" x2="900" y2="210" stroke="#374151" stroke-width="4"/>
                
                <line x1="70" y1="30" x2="70" y2="210" stroke="#1f2937" stroke-dasharray="6 6"/>
                <text x="70" y="24" fill="#6b7280" font-size="12" font-weight="bold" text-anchor="middle">0m (Truck)</text>
                
                <line x1="285" y1="30" x2="285" y2="210" stroke="#1f2937" stroke-dasharray="6 6"/>
                <text x="285" y="24" fill="#6b7280" font-size="12" font-weight="bold" text-anchor="middle">75m</text>

                <line x1="500" y1="30" x2="500" y2="210" stroke="#1f2937" stroke-dasharray="6 6"/>
                <text x="500" y="24" fill="#6b7280" font-size="12" font-weight="bold" text-anchor="middle">150m</text>

                <line x1="715" y1="30" x2="715" y2="210" stroke="#1f2937" stroke-dasharray="6 6"/>
                <text x="715" y="24" fill="#6b7280" font-size="12" font-weight="bold" text-anchor="middle">225m</text>

                <!-- Incline Road Bed Transform Group -->
                <g transform="rotate({incline_angle}, 70, 210)">
                    <!-- Road Bed Surface -->
                    <rect x="0" y="210" width="900" height="30" fill="#1f2937"/>
                    <line x1="0" y1="210" x2="900" y2="210" stroke="#f59e0b" stroke-width="3" stroke-dasharray="12 10"/>

                    <!-- LiDAR Sensor Vision Beam (Cyan Gradient Cone) -->
                    <polygon points="{truck_x + 60},180 {truck_x + 60 + lidar_width},150 {truck_x + 60 + lidar_width},205" fill="rgba(6, 182, 212, 0.20)" stroke="#06b6d4" stroke-width="1.5" stroke-dasharray="4 4"/>

                    <!-- Required Safe Stopping Envelope (Red/Amber Bar) -->
                    <rect x="{truck_x + 65}" y="200" width="{stopping_width}" height="10" fill="{alert_border}" rx="5" opacity="0.9"/>
                    <text x="{truck_x + 65 + (stopping_width/2)}" y="193" fill="{alert_text}" font-size="12" font-weight="800" text-anchor="middle">Req Stop Distance: {safe_stopping_distance:.1f}m</text>

                    <!-- Haul Truck (Heavy Mine Vehicle Graphic) -->
                    <g transform="translate({truck_x}, 150)">
                        <rect x="0" y="12" width="65" height="48" fill="#f59e0b" rx="6" stroke="#d97706" stroke-width="2.5"/>
                        <rect x="46" y="0" width="24" height="34" fill="#111827" rx="4" stroke="#f59e0b" stroke-width="2"/>
                        <circle cx="16" cy="60" r="10" fill="#030712" stroke="#e5e7eb" stroke-width="4"/>
                        <circle cx="50" cy="60" r="10" fill="#030712" stroke="#e5e7eb" stroke-width="4"/>
                        <text x="32.5" y="38" fill="#030712" font-size="11" font-weight="900" text-anchor="middle">HAUL TRUCK</text>
                    </g>

                    <!-- Obstacle Vehicle / Mine Barrier Ahead -->
                    <g transform="translate({obstacle_x}, 145)">
                        <rect x="0" y="5" width="40" height="58" fill="#ef4444" rx="6" stroke="#991b1b" stroke-width="2.5"/>
                        <line x1="0" y1="5" x2="40" y2="63" stroke="#7f1d1d" stroke-width="3"/>
                        <line x1="40" y1="5" x2="0" y2="63" stroke="#7f1d1d" stroke-width="3"/>
                        <text x="20" y="-10" fill="#fca5a5" font-size="13" font-weight="bold" text-anchor="middle">OBSTACLE ({obstacle_distance:.1f}m)</text>
                    </g>
                </g>

                <!-- Atmospheric Fog Layer Overlay -->
                <rect x="0" y="0" width="900" height="270" fill="#9ca3af" opacity="{fog_opacity}"/>

                <!-- Weather Telemetry Overlay Box -->
                <rect x="660" y="16" width="225" height="58" rx="8" fill="#030712" opacity="0.88" stroke="#1f2937"/>
                <text x="772.5" y="36" fill="#f3f4f6" font-size="13" font-weight="bold" text-anchor="middle">Fog Density: {fog_density:.1f}%</text>
                <text x="772.5" y="56" fill="#38bdf8" font-size="12" font-weight="semibold" text-anchor="middle">LiDAR Vision: {lidar_range:.1f}m</text>
            </svg>
        </div>

        <!-- Telemetry Summary Cards Row -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 20px;">
            <div style="background-color: #111827; padding: 14px; border-radius: 10px; border: 1px solid #1f2937; text-align: center;">
                <div style="font-size: 0.8rem; color: #9ca3af; font-weight: 700; text-transform: uppercase;">Vehicle Velocity</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #f9fafb; margin-top: 4px;">{truck_velocity:.1f} <span style="font-size: 0.85rem; color: #9ca3af;">km/h</span></div>
            </div>
            <div style="background-color: #111827; padding: 14px; border-radius: 10px; border: 1px solid #1f2937; text-align: center;">
                <div style="font-size: 0.8rem; color: #9ca3af; font-weight: 700; text-transform: uppercase;">Ramp Incline</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {'#ef4444' if ramp_incline < -5 else '#f9fafb'}; margin-top: 4px;">{ramp_incline:.1f}°</div>
            </div>
            <div style="background-color: #111827; padding: 14px; border-radius: 10px; border: 1px solid #1f2937; text-align: center;">
                <div style="font-size: 0.8rem; color: #9ca3af; font-weight: 700; text-transform: uppercase;">Hazard Profile</div>
                <div style="font-size: 0.95rem; font-weight: 800; color: #38bdf8; margin-top: 6px;">{cluster_name}</div>
            </div>
            <div style="background-color: #111827; padding: 14px; border-radius: 10px; border: 1px solid #1f2937; text-align: center;">
                <div style="font-size: 0.8rem; color: #9ca3af; font-weight: 700; text-transform: uppercase;">Advisory Speed Limit</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #34d399; margin-top: 4px;">{rec_speed:.1f} <span style="font-size: 0.85rem; color: #9ca3af;">km/h</span></div>
            </div>
        </div>

        <!-- Dynamic Safety Directive Banner -->
        <div style="margin-top: 18px; background-color: {alert_bg}; border-left: 6px solid {alert_border}; padding: 16px 20px; border-radius: 8px; color: {alert_text}; font-size: 1.05rem; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
            ⚠️ <strong>Safety Advisory:</strong> {advisory}
        </div>
    </div>
    """
    return svg_html
