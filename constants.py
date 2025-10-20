"""
Constants and error messages for Hunyuan3D API server.
"""

# Error messages
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

# API metadata
API_TITLE = "Hunyuan3D API Server"
API_DESCRIPTION = """
# Hunyuan3D 2.1 API Server

This API server provides endpoints for generating 3D models from 2D images using the Hunyuan3D model.

## Features
TODO: fill it

## Usage

1. Use `/generate` for immediate 3D model generation from images
2. Use `/send` for asynchronous processing with status tracking
3. Use `/status/{uid}` to check task progress and retrieve results
4. Use `/health` to verify service status

## Model Information

- **Model**: Hunyuan3D-2.1 by Tencent
- **License**: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
- **Capabilities**: Image-to-3D, Texture Generation
"""
API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "Hunyuan3D Team",
    "url": "https://github.com/Tencent/Hunyuan3D",
}
API_LICENSE_INFO = {
    "name": "TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT",
    "url": "https://github.com/Tencent/Hunyuan3D/blob/main/LICENSE",
}

# API tags metadata
API_TAGS_METADATA = [
    {
        "name": "generation",
        "description": "3D model generation endpoints. Generate 3D models from 2D images.",
    },
    {
        "name": "status",
        "description": "Health check endpoint. Monitor service health.",
    },
]