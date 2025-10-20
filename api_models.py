from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    worker_id: str = Field(..., description="Worker identifier")
