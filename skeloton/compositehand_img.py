import cv2
import glob
import numpy as np
import os

# 이미지 파일 경로 가져오기
image_files = []
for i in range(0, 132, 3):  # 이미지의 숫자 범위에 맞게 반복문 설정
    left_image_path = f"./controlnet/res_hand_pose_image/pose_left{i:03d}_rgb.png"
    right_image_path = f"./controlnet/res_hand_pose_image/pose_right{i:03d}_rgb.png"

    if os.path.exists(left_image_path) and os.path.exists(right_image_path):
        image_files.append((left_image_path, right_image_path))

# 이미지 합성
for idx, (left_image_path, right_image_path) in enumerate(image_files):
    image1 = cv2.imread(left_image_path)
    image2 = cv2.imread(right_image_path)

    # 이미지 합성
    alpha = 1
    beta = 1
    composite_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

    # 합성된 이미지 저장
    output_path = f"./controlnet/res_hand_pose_image/pose_both_{idx*3:03d}_rgb.png"
    cv2.imwrite(output_path, composite_image)

    print("합성된 이미지 저장 완료:", output_path)
