import numpy as np
import json
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# jason file 불러오기


def face_keypoints(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    # data_json = data['people']['face_keypoints_2d']
    keypoints = np.array(data["people"]["face_keypoints_2d"]).reshape(-1, 3)
    return keypoints


## matplot으로 그려보기
# plt.scatter(keypoints[:, 0], keypoints[:, 1])
# # 키포인트 인덱스를 추가하여 어떤 키포인트가 어디에 있는지 확인합니다.
# for i, (x, y) in enumerate(zip(keypoints[:, 0], keypoints[:, 1])):
#     plt.text(x, y, str(i), fontsize=10, ha='right')

# plt.gca().invert_yaxis()

# 0~16 # 턱선
# 17~21 오른 눈썹
# 22~26 왼 눈썹
# 27~30 코선
# 31~35 코아래선
# 36~41 오른쪽 눈
# 42~47 왼쪽눈
# 48~59 겉입술
# 60~~67 속입술


def draw_face(keypoints, canvas):
    for i in range(len(keypoints)):
        x, y, c = keypoints[i]
        # print(int(x), int(y))
        cv2.circle(canvas, (int(x), int(y)), 4, [225, 225, 225], thickness=-1)
    return canvas


# 그리고 저장하기
# Load json data
# json_path = "./hand/hand/03_real_word_keypoint/NIA_SL_WORD0006_REAL03_F/NIA_SL_WORD0006_REAL03_F_000000000"  # {i:03d}_keypoints.json"
json_path = "./hand/hand/10_real_word_keypoint/NIA_SL_WORD0022_REAL10_F/NIA_SL_WORD0022_REAL10_F_000000000"
filename = []
for i in range(0, 132, 3):
    filename.append(json_path + f"{i:03d}_keypoints.json")
print(filename)

for file in filename:
    data = face_keypoints(file)

    # canvas 설정
    canvas = np.zeros((1280, 2000, 3), dtype=np.uint8)

    # Draw the face pose on the canvas
    canvas = draw_face(data, canvas)
    # 저장하기
    cv2.imwrite(
        f"./controlnet/res_face_pose_image/pose_face_{file[-18:-15]}.png", canvas
    )
