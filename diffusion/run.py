from utils import *
from model import *
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description='make_movie_with_controlnet')
parser.add_argument('--folder_name', type=str,
                    default='./open_pose_images',
                    help='open_pose_image가 들어있는 folder의 이름')

parser.add_argument('--batch_size', type=int,
                    default=2,
                    help='gpu에 맞는 batchsize를 사용할 것')

parser.add_argument('--img_size', type=tuple,
                    default=(400, 512),
                    help='gpu에 맞는 batchsize를 사용할 것')





args = parser.parse_args()


def main(args=args):
    
    #image = get_an_image_resize('./IU.png', args.img_size)
    pose_images = get_images_resize(args.folder_name, args.img_size)
    pose_images = split_by_batch_size(pose_images, args.batch_size)
    for i in range(len(pose_images)):
        output_images = run_control_net(prompt=['positive- delicate, (best quality:1.2), (masterpiece:1.3), realistic, 8K UHD, High definition, High quality texture, intricate details, detailed texture, finely detailed, high detail, extremely detailed cg, High quality shadow, a realistic representation of the face, beautiful detailed, (high detailed skin, skin details), Detailed beautiful delicate face, Detailed beautiful delicate eyes, a face of perfect proportion, Depth of field, Cinematic Light, Lens Flare, Ray tracing, perspective, 20s, (photorealistic:0.8), (solo:1.4), Prominent Nose, (high nose:1.0),(sharp nose:1.0), slender face, (big eyes:1.1),Glow Eyes, blush, glossy lips, perfect body, perfect breasts, (korean celebrity, korean beauty, korean actress, kpop idol, korean), a beautiful 26 years old beautiful korean woman, best ratio four finger and one thumb, (full body:1.5), (makeup:0.4), best reality, high heels, (pantyhose:1.3), long leg , (blonde colored short hair:1.4), (school uniform of korean style:1.2), (shirt + minskirt:1.4), beautiful leg, (view from front:1.5), (smile:1.3), (hand on own waist:1.2), (perfect anatomy:1.2), wind blow    ']*args.batch_size,
                    #person_image=[image]*args.batch_size,
                    control_net_image=pose_images[i],
                    negative_prompt=None,
                    num_inference_steps=30,
                    pipe=available_control_net())
        for j, img in enumerate(output_images):
            output_images.images[j].save(f'./final_images/{i}_{j}.png', 'png')
            
            
        
        
        
            
            
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(args)