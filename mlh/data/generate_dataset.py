import os
import numpy as np
import pandas as pd

def generate_telemetry_dataset(num_samples: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a physics-sound multi-sensor dataset mimicking a digital-twin mining framework
    for haul truck operation in fog and low-visibility conditions.
    """
    np.random.seed(random_seed)
    
    # 1. Fog Density (Continuous percentage: 0% to 100%)
    # Mix of uniform and bimodal distribution to simulate variable weather shifts
    fog_density = np.random.uniform(0.0, 100.0, num_samples)
    
    # 2. LiDAR Range (Laser line-of-sight in meters: 10m to 250m)
    # Physically attenuated by fog density with exponential decay and random atmospheric noise
    max_lidar_base = 250.0
    lidar_attenuation = np.exp(-0.015 * fog_density)
    lidar_noise = np.random.normal(0, 5.0, num_samples)
    lidar_range = np.clip(max_lidar_base * lidar_attenuation + lidar_noise, 10.0, 250.0)
    
    # 3. Truck Velocity (Speed in km/h: 0 to 60 km/h)
    # Haul trucks often reduce speed in fog, but drivers may exceed safe limits on inclines
    base_velocity = np.random.uniform(10.0, 60.0, num_samples)
    # Slight correlation: drivers tend to slow down slightly in fog
    velocity_adjustment = -0.15 * fog_density + np.random.normal(0, 6.0, num_samples)
    truck_velocity = np.clip(base_velocity + velocity_adjustment, 5.0, 60.0)
    
    # 4. Ramp Incline (Grade angle in degrees: -12° downhill to +12° uphill)
    ramp_incline = np.random.uniform(-12.0, 12.0, num_samples)
    
    # 5. Obstacle Distance (Distance to nearest vehicle/wall ahead in meters)
    # Depends partially on LiDAR detection range and random obstacle positioning
    obstacle_distance = np.random.uniform(5.0, 220.0, num_samples)
    
    # -------------------------------------------------------------
    # PHYSICS TARGET CALCULATIONS
    # -------------------------------------------------------------
    # Dynamic Road Friction Coefficient mu:
    # Dry mine road ~0.70. Fog dampens road surface, reducing friction down to ~0.35 in 100% fog.
    mu_base = 0.70
    friction_reduction = 0.40 * (fog_density / 100.0) ** 1.5
    friction_noise = np.random.normal(0, 0.02, num_samples)
    road_friction = np.clip(mu_base - friction_reduction + friction_noise, 0.25, 0.75)
    
    # Gravity and Incline Vector physics
    g = 9.81  # m/s^2
    incline_rad = np.radians(ramp_incline)
    
    # Effective Deceleration a (m/s^2):
    # Downhill incline (negative angle) reduces braking efficiency!
    effective_deceleration = g * (road_friction * np.cos(incline_rad) + np.sin(incline_rad))
    effective_deceleration = np.clip(effective_deceleration, 0.5, 9.0)  # Avoid division by zero or negative decel
    
    # Velocity in meters per second (m/s)
    velocity_ms = truck_velocity * (1000.0 / 3600.0)
    
    # Driver & Automated System Reaction Time t_rxn (seconds):
    # Heavy fog increases perception-reaction delay
    reaction_time = 1.2 + 0.008 * fog_density + np.random.normal(0, 0.1, num_samples)
    reaction_time = np.clip(reaction_time, 1.0, 2.5)
    
    # Physics Safe Stopping Distance Formula:
    # d_stop = (v * t_reaction) + (v^2 / (2 * a_effective)) + physics_uncertainty
    stopping_distance_physics = (velocity_ms * reaction_time) + ((velocity_ms ** 2) / (2.0 * effective_deceleration))
    stopping_noise = np.random.normal(0, 1.5, num_samples)
    safe_stopping_distance = np.clip(stopping_distance_physics + stopping_noise, 2.0, 300.0)
    
    # Binary Target: Collision Risk
    # 1 if Obstacle_Distance < Safe_Stopping_Distance + safety margin, else 0
    safety_margin = 3.0  # 3 meters safety buffer
    collision_risk = (obstacle_distance < (safe_stopping_distance + safety_margin)).astype(int)
    
    # Build Pandas DataFrame
    df = pd.DataFrame({
        'Fog_Density': np.round(fog_density, 2),
        'LiDAR_Range': np.round(lidar_range, 2),
        'Truck_Velocity': np.round(truck_velocity, 2),
        'Ramp_Incline': np.round(ramp_incline, 2),
        'Obstacle_Distance': np.round(obstacle_distance, 2),
        'Road_Friction': np.round(road_friction, 3),
        'Safe_Stopping_Distance': np.round(safe_stopping_distance, 2),
        'Collision_Risk': collision_risk
    })
    
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'mine_telemetry.csv')
    print("Generating synthetic mine vehicle telemetry dataset (6,000 rows)...")
    telemetry_df = generate_telemetry_dataset(num_samples=6000, random_seed=42)
    telemetry_df.to_csv(csv_path, index=False)
    print(f"Dataset successfully generated and saved to {csv_path}!")
    print("\nDataset Summary Statistics:")
    print(telemetry_df.describe())
    print("\nCollision Risk Class Distribution:")
    print(telemetry_df['Collision_Risk'].value_counts(normalize=True))
