import uuid
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
import os
import torch
import shutil
import numpy as np
import tempfile

from PIL import Image
from io import BytesIO
import base64
from api_models import BBoxGenerationRequest
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.

    Args:
        image (str): Base64 encoded image string

    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def _postprocess(mesh, file_name, save_dir):
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    final_save_path = os.path.join(save_dir, '%s.glb' % (file_name))
    mesh.export(final_save_path)
    return final_save_path


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-Omni',
                 subfolder='hunyuan3d-omni',
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache',
                 mc_algo='mc',
                 enable_flashvdm=False):
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.mc_algo = mc_algo
        self.enable_flashvdm = enable_flashvdm
        self.compile = compile

        self.pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(model_path=self.model_path, fast_decode=self.enable_flashvdm)

        for file in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, file))

    @torch.inference_mode()
    def generate_bbox(self, request: BBoxGenerationRequest, uid):
        params = request.model_dump()
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image provided")

        if 'bbox' in params:
            bbox = params["bbox"]
        else:
            raise ValueError("No bounding box provided")

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Processing: {uid}")

        # Validate input file exists
        if not image:
            print(f"Warning: Cannot convert image for uid: {uid}, skipping...")
            raise ValueError("Invalid image provided")

        # Prepare bounding box tensor [1, 1, 6] format
        bbox = torch.FloatTensor(bbox).unsqueeze(0).unsqueeze(0).to(self.pipeline.device).to(self.pipeline.dtype)
        print(f"Bounding box shape: {bbox.shape}")

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            tmp.flush()
            # Run inference with bounding box conditioning
            result = self.pipeline(
                image=tmp.name,
                bbox=bbox,
                num_inference_steps=50,  # Number of denoising steps
                octree_resolution=512,  # 3D resolution for octree representation
                mc_level=0,  # Marching cubes iso-level
                guidance_scale=4.5,  # Classifier-free guidance strength
                generator=torch.Generator('cuda').manual_seed(1234),  # Fixed seed for reproducibility
            )

        # Extract results
        mesh = result['shapes'][0][0]  # Generated 3D mesh

        # Generate output filename with bbox coordinates
        bbox_coords = f"{bbox[0][0][0].item()}_{bbox[0][0][1].item()}_{bbox[0][0][2].item()}"
        file_name = f"{uid}_{bbox_coords}"

        # Save outputs
        final_save_path = _postprocess(mesh, file_name, self.save_dir)

        if self.low_vram_mode:
            torch.cuda.empty_cache()

        return final_save_path, uid