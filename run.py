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


from .infer import Removebg, Image2Views, Views2Mesh, Text2Image

class Hunyuan3D1Text2Image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, text, seed, steps):
        text_to_image_model = Text2Image(
            pretrain=f"{hunyuan3d_path}/weights/hunyuanDiT",
            device="cuda:0",
            save_memory=False
        )

        res_rgb_pil = text_to_image_model(
            text,
            seed=seed,
            steps=steps
        )

        res_rgb_pil.save(os.path.join(output_path, "img.jpg"))

        rembg_model = Removebg()

        res_rgba_pil = rembg_model(res_rgb_pil)

        res_rgb_pil.save(os.path.join(output_path, "img_nobg.png"))

        return res_rgb_pil,

class Hunyuan3D1Test:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ip_adapter": ([True, False],),
                "plus_model": ([True, False],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "run"

    CATEGORY = "Hunyuan3D"

    def run(self, ip_adapter, plus_model):
        test = "test"

        views_to_mesh_model = Views2Mesh(f'{hunyuan3d_path}/svrm/configs/svrm.yaml', f'{hunyuan3d_path}/weights/svrm/svrm.safetensors', "cuda:0", use_lite=False)

        text_to_image_model = Text2Image(
            pretrain=f"{hunyuan3d_path}/weights/hunyuanDiT",
            device="cuda:0",
            save_memory=False
        )

        res_rgb_pil = text_to_image_model(
            "a lovely rabbit eating carrots",
            seed=0,
            steps=20
        )

        res_rgb_pil.save(os.path.join(output_path, "img.jpg"))

        rembg_model = Removebg()

        res_rgba_pil = rembg_model(res_rgb_pil)

        res_rgb_pil.save(os.path.join(output_path, "img_nobg.png"))

        image_to_views_model = Image2Views(device="cuda:0", use_lite=False)

        (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
            res_rgba_pil,
            seed=0,
            steps=50
        )

        views_grid_pil.save(os.path.join(output_path, "views.jpg"))

        views_to_mesh_model(
            views_grid_pil,
            cond_img,
            seed=0,
            target_face_count=90000,
            save_folder=output_path,
            do_texture_mapping=False
        )

        return (test,)

NODE_CLASS_MAPPINGS = {
    "Hunyuan3D V1 Text2Image": Hunyuan3D1Text2Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D V1 Text2Image": Hunyuan3D1Text2Image
}
