import cv2
import os
import imageio


# 이미지 파일 경로
image_folder = "./res_total_pose/"
# GIF 파일 경로
output_gif = "./output.gif"

# 이미지 파일 리스트 가져오기
images_file = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images_file.sort()
folder = []
for image in images_file:
    image_path = os.path.join(image_folder, image)
    folder.append(image_path)
# 이미지를 GIF로 변환

images = [imageio.imread(filename) for filename in folder]
imageio.mimsave("./output.mp4", images, "MP4", duration=1.0)
