import argparse
import asyncio
import uuid
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from api_models import HealthResponse, BBoxGenerationRequest
from model_worker import ModelWorker

from constants import (
    SERVER_ERROR_MSG, DEFAULT_SAVE_DIR, API_TITLE, API_DESCRIPTION,
    API_VERSION, API_CONTACT, API_LICENSE_INFO, API_TAGS_METADATA
)

worker_id = str(uuid.uuid4())[:6]

worker = None
model_semaphore = None

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    openapi_tags=API_TAGS_METADATA
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate", tags=["generation"])
async def generate_3d_model(request: BBoxGenerationRequest):

    uid = uuid.uuid4()

    try:
        file_path, uid = worker.generate_bbox(request, uid)
        return FileResponse(file_path)
    except Exception as e:
        print(f"Error dupa: {e}")
        return JSONResponse(content={"error": SERVER_ERROR_MSG}, status_code=500)

@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    return JSONResponse(content={"status": "healthy", "worker_id": worker_id}, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-Omni')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-sit-Omni')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_flashvdm', action='store_true', help='Use FlashVDM for faster decoding')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--cache-path', type=str, default='./gradio_cache')
    args = parser.parse_args()
    print(f"Starting server with args: {args}")

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    worker = ModelWorker(
        model_path=args.model_path,
        device=args.device,
        low_vram_mode=args.low_vram_mode,
        worker_id=worker_id,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR,
        mc_algo='mc',
        enable_flashvdm=args.enable_flashvdm
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")