from diffusers.models import AutoencoderKL
import diffusers
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerDiscreteScheduler, StableDiffusionPipeline
import torch

class Diff_Run:
    def __init__(self, 
                 prompt,
                 model_name = 'Lykon/DreamShaper',
                 negative_prompt = "artoon, 3d, ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((b&w)), weird colors, blurry",
                 cfg = 6,
                 num_inference_steps = 30):
        self.model_name = model_name
        self.prompt = prompt
        self.negtive_prompt = negative_prompt
        self.cfg = cfg
        self.num_inference_steps = num_inference_steps
        
        
    def controlnet(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",
                                                    torch_dtype=torch.float16,
                                                    safety_checker = None,
                                                    requires_safety_checker = False)
        return controlnet
    
    
    def text2imgpipe(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe
    
    def text2img_controlnetpipe(self):
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_name, controlnet=self.controlnet, torch_dtype=torch.float16
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe
    
    def text2img_lorapipe(self, model_path='./jwy.safetensors'):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.unet.load_attn_procs(model_path)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe
    
    def text2img(self,
                pipe,
                ):
                

        diffusion_images = pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=6, # 20, 7.5 # cfg
            generator=torch.Generator(device="cuda").manual_seed(112),
            num_inference_steps=self.num_inference_steps
        )
        return diffusion_images

    def text2img_contronetl(self,
                pipe,
                control_net_image,
                ):

        diffusion_images = pipe(
            control_net_image,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=6, # 20, 7.5 # cfg # prompt를 얼마나 줄지 
            controlnet_conditioning_scale=1.8, # controlnet강도 
            generator=torch.Generator(device="cuda").manual_seed(112),
            num_inference_steps=self.num_inference_steps
        )
        return diffusion_images

    def text2img_lora(self,
                pipe,
                ):
                
        if negative_prompt is None:
            negative_prompt = ["artoon, 3d, ((disfigured)), ((bad art)), ((deformed)), ((poorly drawn)), ((extra limbs)), ((b&w)), weird colors, blurry"]

        diffusion_images = pipe(
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            guidance_scale=self.cfg,
            generator=torch.Generator(device="cuda").manual_seed(112),
            num_inference_steps=self.num_inference_steps
        )
        return diffusion_images



