import os
import sys
import folder_paths
import torch

comfy_path = os.path.dirname(folder_paths.__file__)

hunyuan3d_path = f'{comfy_path}/custom_nodes/ComfyUI-Hunyuan3D-1-wrapper'

output_path = f'{comfy_path}/output/Hunyuan3D-1/'

os.makedirs(output_path, exist_ok=True)

sys.path.append(hunyuan3d_path)

python_executable_path = sys.executable
print("python_executable_path: ", python_executable_path)

python_embeded_path = os.path.dirname(python_executable_path)

print("python_embeded_path: ", python_embeded_path)

python_scripts_path = os.path.join(python_embeded_path, 'Scripts')

os.environ['PATH'] = os.environ['PATH'] + ";" + python_scripts_path

from .infer import Text2Image, Text2ImagePipelineLoad, Image2Views, Image2ViewsPipelineLoad, Removebg, Views2Mesh, \
    Views2MeshPipelineLoad, GifRenderer

from PIL import Image
import numpy as np

class Hunyuan3D1Text2ImagePipelineLoad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_memory": ([True, False],),
            }
        }

    RETURN_TYPES = ("Hunyuan3D1Text2ImagePipelineConfig",)
    RETURN_NAMES = ("pipeline_config",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, save_memory):
        pipeline_config = Text2ImagePipelineLoad(
            pretrain=f"{hunyuan3d_path}/weights/hunyuanDiT",
            device="cuda:0",
            save_memory=save_memory
        )

        return pipeline_config,


class Hunyuan3D1Image2ViewsPipelineLoad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_lite": ([True, False],),
            }
        }

    RETURN_TYPES = ("Hunyuan3D1Image2ViewsPipelineConfig",)
    RETURN_NAMES = ("pipeline_config",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, use_lite):
        pipeline_config = Image2ViewsPipelineLoad(use_lite=use_lite)

        return pipeline_config,


class Hunyuan3D1Views2MeshPipelineLoad:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_lite": ([True, False],),
            }
        }

    RETURN_TYPES = ("Hunyuan3D1Views2MeshPipelineConfig",)
    RETURN_NAMES = ("pipeline_config",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, use_lite):
        pipeline_config = Views2MeshPipelineLoad(f'{hunyuan3d_path}/svrm/configs/svrm.yaml',
                                                 f'{hunyuan3d_path}/weights/svrm/svrm.safetensors', "cuda:0",
                                                 use_lite=use_lite)

        return pipeline_config,

class Hunyuan3D1ImageLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("Hunyuan3D1Image",)
    RETURN_NAMES = ("output",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, image):
        img_batch_np = image.cpu().detach().numpy().__mul__(255.).astype(np.uint8)

        res_rgb_pil = Image.fromarray(img_batch_np[0])

        rembg_model = Removebg()

        res_rgba_pil = rembg_model(res_rgb_pil)

        res_rgb_pil.save(os.path.join(output_path, "img_nobg.png"))

        return res_rgba_pil,

class Hunyuan3D1Text2Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline_config": ("Hunyuan3D1Text2ImagePipelineConfig",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
            },
        }

    RETURN_TYPES = ("Hunyuan3D1Image",)
    RETURN_NAMES = ("output",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, pipeline_config, text, seed, steps):
        text_to_image_model = Text2Image(pipeline_config)

        res_rgb_pil = text_to_image_model(
            text,
            seed=seed,
            steps=steps
        )

        res_rgb_pil.save(os.path.join(output_path, "img.jpg"))

        rembg_model = Removebg()

        res_rgba_pil = rembg_model(res_rgb_pil)

        res_rgb_pil.save(os.path.join(output_path, "img_nobg.png"))

        return res_rgba_pil,


class Hunyuan3D1Image2Views:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("Hunyuan3D1Image",),
                "pipeline_config": ("Hunyuan3D1Image2ViewsPipelineConfig",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
            }
        }

    RETURN_TYPES = ("Hunyuan3D1ViewGridPil", "Hunyuan3D1CondImage",)
    RETURN_NAMES = ("views_grid_pil", "cond_img",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, input, pipeline_config, seed, steps):
        image_to_views_model = Image2Views(pipeline_config)

        (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
            input,
            seed=seed,
            steps=steps
        )

        views_grid_pil.save(os.path.join(output_path, "views.jpg"))

        return views_grid_pil, cond_img,


class Hunyuan3D1Views2Mesh:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "views_grid_pil": ("Hunyuan3D1ViewGridPil",),
                "cond_img": ("Hunyuan3D1CondImage",),
                "pipeline_config": ("Hunyuan3D1Views2MeshPipelineConfig",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "target_face_count": ("INT", {"default": 90000, "min": 10000, "max": 100000}),
                #"do_texture_mapping": ([True, False],), disable for now
                #"do_render": ([True, False],), disable for now
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, views_grid_pil, cond_img, pipeline_config, seed, target_face_count):
        views_to_mesh_model = Views2Mesh(pipeline_config)

        views_to_mesh_model(
            views_grid_pil,
            cond_img,
            seed=seed,
            target_face_count=target_face_count,
            save_folder=output_path,
            do_texture_mapping=False # disable texture mapping for now
        )

        # if do_render: #disable for now
        if False:
            gif_renderer = GifRenderer()

            gif_renderer(
                os.path.join(output_path, 'mesh.obj'),
                gif_dst_path=os.path.join(output_path, 'output.gif'),
            )

        return views_grid_pil,


NODE_CLASS_MAPPINGS = {
    "Hunyuan3D V1 - Image Loader": Hunyuan3D1ImageLoader,
    "Hunyuan3D V1 - Text2Image": Hunyuan3D1Text2Image,
    "Hunyuan3D V1 - Text2Image Pipeline Load": Hunyuan3D1Text2ImagePipelineLoad,
    "Hunyuan3D V1 - Image2Views": Hunyuan3D1Image2Views,
    "Hunyuan3D V1 - Image2Views Pipeline Load": Hunyuan3D1Image2ViewsPipelineLoad,
    "Hunyuan3D V1 - Views2Mesh": Hunyuan3D1Views2Mesh,
    "Hunyuan3D V1 - Views2Mesh Pipeline Load": Hunyuan3D1Views2MeshPipelineLoad
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D V1 - Image Loader": Hunyuan3D1ImageLoader,
    "Hunyuan3D V1 - Text2Image": Hunyuan3D1Text2Image,
    "Hunyuan3D V1 - Text2Image Pipeline Load": Hunyuan3D1Text2ImagePipelineLoad,
    "Hunyuan3D V1 - Image2Views": Hunyuan3D1Image2Views,
    "Hunyuan3D V1 - Image2Views Pipeline Load": Hunyuan3D1Image2ViewsPipelineLoad,
    "Hunyuan3D V1 - Views2Mesh": Hunyuan3D1Views2Mesh,
    "Hunyuan3D V1 - Views2Mesh Pipeline Load": Hunyuan3D1Views2MeshPipelineLoad
}
