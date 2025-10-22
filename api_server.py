import argparse
import uuid
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_models import HealthResponse
from model_worker import ModelWorker

from constants import (
    SERVER_ERROR_MSG, API_TITLE, API_DESCRIPTION,
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

from fastapi import UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import json, io, uuid, asyncio


@app.post("/generate", tags=["generation"])
async def generate_3d_model(
    request: Request,
    control_type: str = Form(...),
    image: UploadFile = File(...),
    bbox: str | None = Form(None),
    pointcloud: UploadFile | None = File(None),
):
    """
    Unified /generate endpoint supporting bbox, voxel, point, and pose inputs.
    For control_type == 'pose', accepts any number of text form fields dynamically.
    """

    if worker.model_semaphore.locked():
        return JSONResponse(
            {"error": "Model is busy, please try again later"},
            status_code=429,
        )

    uid = str(uuid.uuid4())

    # ---- load and preprocess the image ----
    try:
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Invalid image: {e}"}, status_code=400)

    try:
        lowered_control_type = control_type.lower()
        if lowered_control_type == "bbox":
            if not bbox:
                return JSONResponse({"error": "bbox data required"}, status_code=400)
            bbox_vals = json.loads(bbox)
            file_path, uid = await worker.generate_bbox(img, bbox_vals, uid)

        elif lowered_control_type == "voxel":
            if not pointcloud:
                return JSONResponse({"error": "pointcloud file required"}, status_code=400)
            cloud_bytes = await pointcloud.read()
            file_path, uid = await worker.generate_voxel(img, cloud_bytes, uid)

        elif lowered_control_type == "point":
            if not pointcloud:
                return JSONResponse({"error": "pointcloud file required"}, status_code=400)
            cloud_bytes = await pointcloud.read()
            file_path, uid = await worker.generate_point(img, cloud_bytes, uid)

        elif lowered_control_type == "pose":
            # collect all form fields except control_type, image, bbox, pointcloud
            form_data = await request.form()
            pose_cfg = {
                k: v
                for k, v in form_data.items()
                if isinstance(v, str) and k not in ("control_type", "bbox")
            }
            if not pose_cfg:
                return JSONResponse({"error": "No pose description provided"}, status_code=400)
            file_path, uid = await worker.generate_pose(img, pose_cfg, uid)

        else:
            return JSONResponse(
                {"error": f"Unknown control_type '{control_type}'"},
                status_code=400,
            )

        return FileResponse(file_path)

    except Exception as e:
        print(f"Generation error: {e}")
        return JSONResponse({"error": SERVER_ERROR_MSG}, status_code=500)


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    return JSONResponse(content={"status": "healthy", "worker_id": worker_id}, status_code=200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP address")
    parser.add_argument("--port", type=int, default=8081, help="Host port")
    parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-Omni", help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--limit-model-concurrency", type=int, default=2, help="Limit the number of concurrent model runs")
    parser.add_argument("--enable_flashvdm", action="store_true", help="Use FlashVDM for faster decoding")
    parser.add_argument("--low_vram_mode", action="store_true", help="Empty cuda cache after each model run to reduce VRAM usage.")
    parser.add_argument("--cache-path", type=str, default="./gradio_cache", help="Path to store cached models.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for model initialization.")
    parser.add_argument("--clean_cache", action="store_true", help="Clean cache folder before starting the server.")
    args = parser.parse_args()
    print(f"Starting server with args: {args}")

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    worker = ModelWorker(
        model_path=args.model_path,
        device=args.device,
        low_vram_mode=args.low_vram_mode,
        model_semaphore=model_semaphore,
        save_dir=SAVE_DIR,
        enable_flashvdm=args.enable_flashvdm,
        seed=args.seed,
        clean_cache=args.clean_cache
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")