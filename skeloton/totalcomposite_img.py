import cv2
import glob
import numpy as np
import os

# 이미지 파일 경로 가져오기
image_files = []
for i in range(0, 132, 3):  # 이미지의 숫자 범위에 맞게 반복문 설정
    hand_image_path = f"./controlnet/res_hand_pose_image/pose_both_{(i):03d}_rgb.png"
    body_image_path = f"./controlnet/res_body_pose_image/pose_body{i:03d}_dot.png"
    face_image_path = f"./controlnet/res_face_pose_image/pose_face_{(i):03d}.png"

    if (
        os.path.exists(hand_image_path)
        and os.path.exists(body_image_path)
        and os.path.exists(face_image_path)
    ):
        image_files.append((hand_image_path, body_image_path, face_image_path))
print(image_files)


# 이미지 크롭
def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty : starty + cropy, startx : startx + cropx]


# 이미지 사이즈 줄이기
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


print(len(image_files))
# 이미지 합성
for idx, (hand_image_path, body_image_path, face_image_path) in enumerate(image_files):
    img1 = cv2.imread(hand_image_path)
    img2 = cv2.imread(body_image_path)
    img3 = cv2.imread(face_image_path)

    # 이미지 합성
    alpha = 1
    beta = 1
    gamma = 1
    composite_image = cv2.addWeighted(img1, alpha, img2, beta, 0)
    composite_image = cv2.addWeighted(composite_image, 1, img3, gamma, 0)

    # # 합성된 이미지 저장
    # output_path = f"./res_total_pose/pose_total_{(idx+1)*10}.png"
    # cv2.imwrite(output_path, composite_image)

    # print("합성된 이미지 저장 완료:", output_path)

    # 이미지 사이즈 줄이기
    resized_img = resize_image(composite_image, 80)
    # 2000 * 1280
    crop_path = (
        f"./controlnet/res_total_pose/pose_total_crop_960x540_{(idx)*3:03d}_new.png"
    )
    # cropped_canvas = crop_center(resized_img, 960, 540)
    # cv2.imwrite(crop_path, cropped_canvas)
    # print("크롭 이미지 저장", crop_path)
    crop_width = 540
    crop_height = 960

    # 이미지 크기 가져오기
    img_height, img_width, _ = resized_img.shape

    # 이미지 자르기
    # start_x = (img_width - crop_width ) // 2
    start_x = (img_width - crop_width - 60) // 2
    start_y = (img_height - crop_height - 60) // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    cropped_image = resized_img[start_y:end_y, start_x:end_x]
    print("크롭 이미지 저장", crop_path)
    cv2.imwrite(crop_path, cropped_image)
