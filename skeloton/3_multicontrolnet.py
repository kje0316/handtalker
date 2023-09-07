import cv2
from PIL import Image
import numpy as np


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained("fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=[controlnet, controlnet], torch_dtype=torch.float16
)

from diffusers import UniPCMultistepScheduler

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()




face_image = load_image("./face_resize.png")
body_image = load_image("./pose_image/pose_body_050.png")

# controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")
prompt = ["best quality, extremely detailed, 1girl"]
seed = 42
image = pipe(
        ["(white backgroud:1.2) , best quality,  1girl"],
        [body_image, face_image],
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
        generator=torch.Generator(device="cpu").manual_seed(seed),
        num_inference_steps=50,
        )




image.images[0].save(f"res_hand_body_{seed}_8.png", 'png')