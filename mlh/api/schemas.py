from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TelemetryInput(BaseModel):
    Fog_Density: float = Field(..., ge=0.0, le=100.0, description="Fog Density continuous percentage (0% to 100%)")
    LiDAR_Range: float = Field(..., ge=5.0, le=300.0, description="LiDAR line-of-sight range in meters")
    Truck_Velocity: float = Field(..., ge=0.0, le=80.0, description="Truck Velocity in km/h")
    Ramp_Incline: float = Field(..., ge=-20.0, le=20.0, description="Ramp Incline grade angle in degrees (-12° to +12°)")
    Obstacle_Distance: float = Field(..., ge=1.0, le=500.0, description="Distance to nearest obstacle ahead in meters")

    class Config:
        json_schema_extra = {
            "example": {
                "Fog_Density": 65.5,
                "LiDAR_Range": 75.0,
                "Truck_Velocity": 45.0,
                "Ramp_Incline": -6.0,
                "Obstacle_Distance": 30.0
            }
        }

class TelemetryBatchInput(BaseModel):
    items: List[TelemetryInput]

class PredictionOutput(BaseModel):
    safe_stopping_distance_meters: float
    obstacle_distance_meters: float
    collision_risk_flag: int
    collision_risk_probability_percent: float
    hazard_cluster_id: int
    hazard_cluster_name: str
    recommended_max_speed_kmh: float
    advisory_action: str
    model_used: str

class ModelMetricsResponse(BaseModel):
    status: str
    metrics: Dict[str, Any]

class ClusterInfoResponse(BaseModel):
    status: str
    clusters: Dict[str, Any]
