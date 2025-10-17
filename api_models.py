from pydantic import BaseModel, Field
from typing import Optional


class BBoxGenerationRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded input image for 3D generation")

    bbox: list[float] = Field(..., description="Bounding box coordinates for 3D generation")

    # octree_resolution: int = Field(
    #     256,
    #     description="Resolution of the octree for mesh generation",
    #     ge=64,
    #     le=512
    # )


class GenerationResponse(BaseModel):
    """Response model for generation status"""
    uid: str = Field(..., description="Unique identifier for the generation task")


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str = Field(..., description="Status of the generation task")
    model_base64: Optional[str] = Field(
        None,
        description="Base64 encoded generated model file (only when status is 'completed')"
    )
    message: Optional[str] = Field(
        None,
        description="Error message (only when status is 'error')"
    )


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    worker_id: str = Field(..., description="Worker identifier")
