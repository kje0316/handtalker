import cv2
import numpy as np
import os
from PIL import Image


def get_an_image_resize(img_dir, hw):
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)
    img = cv2.resize(img, hw)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
            
    return img


# folder dirs에서 image 갖고오기 # 갖고오면 resize도 같이 해줌.
def get_images_resize(folder_dir, hw):
    img_dirs = os.listdir(folder_dir)
        
    imgs = []
    for img in sorted(img_dirs):
        if img.endswith(".png") or img.endswith(".jpg"):
            img = os.path.join(folder_dir, img)
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, hw)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgs.append(img)
            
    return imgs

# batchsize로 나눠주고 남은 image는 삭제
# ex) 18장의 image를 받고 batchsize가 4이면 뒤에 마지막 두장을 삭제하고 [[4장],[4장],[4장],[4장]] 다음과 같은 리스트 형식으로 리턴해줌.
def split_by_batch_size(imgs, batch_size):
    imgs_cnt = int((len(imgs) // batch_size) * batch_size)
    imgs = imgs[:imgs_cnt]
    imgs = [imgs[batch_size * i : batch_size * (i+1)] for i in range(len(imgs)//batch_size)]
    return imgs



    
    




        
    