# Open Source Model Licensed under the Apache License Version 2.0 and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 
# except for the third-party components listed below. 
# Hunyuan 3D does not impose any additional limitations beyond what is outlined 
# in the repsective licenses of these third-party components. 
# Users must comply with all terms and conditions of original licenses of these third-party 
# components and must ensure that the usage of the third party components adheres to 
# all relevant laws and regulations. 

# For avoidance of doubts, Hunyuan 3D means the large language models and 
# their software and algorithms, including trained model weights, parameters (including 
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code, 
# fine-tuning enabling code and other elements of the foregoing made publicly available 
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import torch
import random
import numpy as np
from PIL import Image
from einops import rearrange
from PIL import Image, ImageSequence

from .utils import seed_everything, timing_decorator, auto_amp_inference
from .utils import get_parameter_number, set_parameter_grad_false
from mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from mvd.hunyuan3d_mvd_lite_pipeline import Hunyuan3d_MVD_Lite_Pipeline

import os
import folder_paths

comfy_path = os.path.dirname(folder_paths.__file__)

hunyuan3d_path = f'{comfy_path}/custom_nodes/ComfyUI-Hunyuan3D-1-wrapper'

class Image2ViewsPipelineLoad():
    def __init__(self, device="cuda:0", use_lite=False):
        """
        初始化并加载图像转视图的模型管道
        Args:
            device: 设备类型
            use_lite: 是否使用轻量版模型
        """
        self.device = device
        if use_lite:
            self.pipe = Hunyuan3d_MVD_Lite_Pipeline.from_pretrained(
                f'{hunyuan3d_path}/weights/mvd_lite',
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else:
            self.pipe = HunYuan3D_MVD_Std_Pipeline.from_pretrained(
                f'{hunyuan3d_path}/weights/mvd_std',
                torch_dtype=torch.float16,
                use_safetensors=True,
            )

        self.pipe = self.pipe.to(device)
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]
        set_parameter_grad_false(self.pipe.unet)
        print('image2views unet model', get_parameter_number(self.pipe.unet))

    def get_pipeline_config(self):
        """返回管道配置"""
        return {
            'pipe': self.pipe,
            'device': self.device,
            'order': self.order
        }
        