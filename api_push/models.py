from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime

class APIResponse(BaseModel):
    """Standard API response model"""
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PushPayload(BaseModel):
    """Flexible model for incoming push requests"""
    data: Optional[Dict[str, Any]] = Field(None, description="The JSON data to be processed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    uuid: Optional[str] = Field(None, description="Unique record identifier")
    # Allow extra fields like 'detection_results', 'summary', etc.
    class Config:
        extra = "allow"

class BusOccupancyData(BaseModel):
    """Example model for bus occupancy data"""
    bus_id: str = Field(..., description="Unique identifier for the bus")
    route_id: str = Field(..., description="Route identifier")
    occupancy_count: int = Field(..., ge=0, description="Number of passengers")
    max_capacity: int = Field(..., gt=0, description="Maximum capacity of the bus")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
    
    @property
    def occupancy_percentage(self) -> float:
        """Calculate occupancy percentage"""
        return (self.occupancy_count / self.max_capacity) * 100

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    database_connected: bool
