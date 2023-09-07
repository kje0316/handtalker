import cv2
import os

# 이미지 파일 경로
image_folder = "./res_total_pose/"
# 동영상 파일 경로

output_video = "./handtalker15_05.mp4"

# 이미지 파일 리스트 가져오기
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(image_folder, images[0]))

height, width, layers = frame.shape

# 동영상 인코더 설정
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, 15, (width, height))

# 이미지를 동영상으로 변환
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

cv2.destroyAllWindows()
video.release()
