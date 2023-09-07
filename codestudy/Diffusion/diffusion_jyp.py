from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DiffusionPipeline
import torch
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import cv2
import numpy as np

image = load_image(
    "./1.png"
)

canny_image = np.array(image)

print(canny_image.shape)
low_threshold = 100
high_threshold = 200


image = cv2.Canny(canny_image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)


pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()

pipe.enable_xformers_memory_efficient_attention()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

prompt = "masterpiece, best quality, extremely detailed, beautiful"
prompt = [t + prompt for t in ["sunset", "sun rising", "sky", "taylor swift"]]
generator = [torch.Generator(device="cuda").manual_seed(102) for i in range(len(prompt))]

output = pipe(
    prompt,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
    generator=generator,
    num_inference_steps=30,
)

print(output.images[0])

for i in range(4):
    output.images[i].save(f'./images/{i+2}.png', 'png')
