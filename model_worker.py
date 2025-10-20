import asyncio
import trimesh
from inference import normalize_mesh
from hy3dshape.pipelines import Hunyuan3DOmniSiTFlowMatchingPipeline
import io
import os
import torch
import shutil
import numpy as np
import tempfile
from PIL import Image
from io import BytesIO
from hy3dshape.postprocessors import FloaterRemover, DegenerateFaceRemover

def _postprocess(mesh, file_name, save_dir):
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    final_save_path = os.path.join(save_dir, '%s.glb' % (file_name))
    print(f"Saving to {final_save_path}")
    mesh.export(final_save_path)
    return final_save_path


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-Omni',
                 device='cuda',
                 low_vram_mode=False,
                 model_semaphore=None,
                 save_dir='gradio_cache',
                 enable_flashvdm=False,
                 seed=1234,
                 clean_cache=False):
        self.model_path = model_path
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = os.path.abspath(save_dir)
        self.enable_flashvdm = enable_flashvdm
        self.seed = seed
        self.clean_cache = clean_cache

        self.pipeline = Hunyuan3DOmniSiTFlowMatchingPipeline.from_pretrained(model_path=self.model_path, fast_decode=self.enable_flashvdm)

        if self.clean_cache:
            for name in os.listdir(self.save_dir):
                path = os.path.join(self.save_dir, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")

    async def generate_voxel(self, image: Image.Image, point_cloud: bytes, uid: str):
        async with self.model_semaphore:
            final_file_path, uid = await asyncio.to_thread(self._generate_voxel, image, point_cloud, uid)
            return final_file_path, uid

    async def generate_bbox(self, image: Image.Image, bbox: list[float], uid: str):
        async with self.model_semaphore:
            final_file_path, uid = await asyncio.to_thread(self._generate_bbox, image, bbox, uid)
            return final_file_path, uid

    async def generate_point(self, image: Image.Image, point_cloud: bytes, uid: str):
        async with self.model_semaphore:
            final_file_path, uid = await asyncio.to_thread(self._generate_point, image, point_cloud, uid)
            return final_file_path, uid

    async def generate_pose(self, image: Image.Image, pose_config: dict[str, str], uid: str):
        async with self.model_semaphore:
            final_file_path, uid = await asyncio.to_thread(self._generate_pose, image, pose_config, uid)
            return final_file_path, uid

    @torch.inference_mode()
    def _generate_pose(self, image: Image.Image, pose_config: dict[str, str], uid: str):
        os.makedirs(self.save_dir, exist_ok=True)
        archive_dir = os.path.join(self.save_dir, str(uid))
        os.makedirs(archive_dir, exist_ok=True)
        print(f"Processing: {uid}")

        for pose_key in pose_config.keys():
            bone = pose_config[pose_key]

            bone_points = torch.from_numpy(np.loadtxt(io.StringIO(bone))).to(self.pipeline.device).to(self.pipeline.dtype).unsqueeze(
                0)
            print(f"pose: {bone_points.shape}")

            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                image.save(tmp, format="PNG")
                tmp.flush()
                result = self.pipeline(
                    image=tmp.name,
                    pose=bone_points,
                    num_inference_steps=50,
                    octree_resolution=512,
                    mc_level=0,
                    guidance_scale=4.5,
                    generator=torch.Generator(self.device).manual_seed(self.seed),
                )
            mesh = result['shapes'][0][0]
            file_name = f"pose_{uid}_{pose_key}"
            final_save_path = _postprocess(mesh, file_name, archive_dir)

        shutil.make_archive(archive_dir, 'zip', archive_dir)

        if self.low_vram_mode:
            torch.cuda.empty_cache()

        return archive_dir + ".zip", uid

    @torch.inference_mode()
    def _generate_point(self, image: Image.Image, point_cloud: bytes, uid: str):
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Processing: {uid}")

        mesh = trimesh.load(BytesIO(point_cloud), file_type='ply')
        mesh = normalize_mesh(mesh, scale=0.98)
        surface = mesh.vertices
        # surface[:, 2] = surface[:, 2] + 0.3
        surface = torch.FloatTensor(surface).unsqueeze(0)
        surface = surface.to(self.pipeline.device).to(self.pipeline.dtype)

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            tmp.flush()
            result = self.pipeline(
                image=tmp.name,
                point=surface,
                num_inference_steps=50,
                octree_resolution=512,
                mc_level=0,
                guidance_scale=4.5,
                generator=torch.Generator(self.device).manual_seed(self.seed),
            )

        mesh = result['shapes'][0][0]  # [0]

        file_name = f"point_{uid}"
        final_save_path = _postprocess(mesh, file_name, self.save_dir)

        if self.low_vram_mode:
            torch.cuda.empty_cache()

        return final_save_path, uid

    @torch.inference_mode()
    def _generate_voxel(self, image: Image.Image, point_cloud: bytes, uid: str):
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Processing: {uid}")

        mesh = trimesh.load(BytesIO(point_cloud), file_type='ply')
        rotation_matrix = trimesh.transformations.rotation_matrix(
            angle=np.radians(-90),
            direction=[1, 0, 0])
        mesh.apply_transform(rotation_matrix)
        mesh = normalize_mesh(mesh)
        surface = mesh.sample(81920)
        surface = torch.FloatTensor(surface).unsqueeze(0)
        surface = surface.to(self.pipeline.device).to(self.pipeline.dtype)
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            tmp.flush()
            # Run inference with bounding box conditioning
            result = self.pipeline(
                image=tmp.name,
                point=surface,
                num_inference_steps=50,
                octree_resolution=512,
                mc_level=0,
                guidance_scale=4.5,
                generator=torch.Generator(self.device).manual_seed(self.seed),
            )

        mesh = result['shapes'][0][0]  # [0]
        file_name = f"voxel_{uid}"
        final_save_path = _postprocess(mesh, file_name, self.save_dir)

        if self.low_vram_mode:
            torch.cuda.empty_cache()

        return final_save_path, uid

    @torch.inference_mode()
    def _generate_bbox(self, image: Image.Image, bbox: list[float], uid: str):

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Processing: {uid}")

        # Validate input file exists
        if not image:
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
                generator=torch.Generator(self.device).manual_seed(self.seed),  # Fixed seed for reproducibility
            )

        # Extract results
        mesh = result['shapes'][0][0]  # Generated 3D mesh

        # Generate output filename with bbox coordinates
        bbox_coords = f"{bbox[0][0][0].item()}_{bbox[0][0][1].item()}_{bbox[0][0][2].item()}"
        file_name = f"bbox_{uid}_{bbox_coords}"

        # Save outputs
        final_save_path = _postprocess(mesh, file_name, self.save_dir)

        if self.low_vram_mode:
            torch.cuda.empty_cache()

        return final_save_path, uid