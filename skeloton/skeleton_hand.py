import numpy as np
import json
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# jason file 불러오기
# file_name = '/content/drive/MyDrive/Colab Notebooks/수어_영상_생성_서비스_프로젝트/dataset/NIA_SL_WORD1507_REAL01_F_000000000050_keypoints.json'
def json2keypoints(json_file, hand_dir):
    with open(json_file, "r") as file:
        data = json.load(file)
    # data_json = data['people']['hand_left_keypoints_2d']
    keypoints = np.array(data["people"][f"hand_{hand_dir}_keypoints_2d"]).reshape(-1, 3)
    return keypoints


# ## matplot으로 그려서 확인 하기
# keypoints = json2keypoints(file_name, 'right')
# plt.scatter(keypoints[:, 0], keypoints[:, 1])
# # 키포인트 인덱스를 추가하여 어떤 키포인트가 어디에 있는지 확인합니다.
# for i, (x, y) in enumerate(zip(keypoints[:, 0], keypoints[:, 1])):
#     plt.text(x, y, str(i), fontsize=10, ha='right')

# plt.gca().invert_yaxis()

# 색상 값확인
import cv2

rgb_colors = ["BE280E", "A1C900", "00C73C", "22519E", "B300CD"]
bgr_colors = []

for rgb_color in rgb_colors:
    rgb_tuple = tuple(int(rgb_color[i : i + 2], 16) for i in (0, 2, 4))
    bgr_color = [rgb_tuple[2], rgb_tuple[1], rgb_tuple[0]]
    bgr_colors.append(bgr_color)

# print(bgr_colors) rbg
# [[14, 40, 190], [0, 201, 161], [60, 199, 0], [158, 81, 34], [205, 0, 179]]


def draw_hand_right(json_data, canvas):
    limbSeq = [
        [0, 1],
        [1, 2],
        [2, 3],
        # [3, 4],  # 엄지
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],  # 검지
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],  # 중지
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],  # 약지
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    # joint_color = [
    #     [14, 40, 190],
    #     [0, 201, 161],
    #     [60, 199, 0],
    #     [158, 81, 34],
    #     [205, 0, 179],
    # ]
    limSeq_color = [
        [14, 40, 190],
        [14, 40, 190],
        [14, 40, 190],
        [14, 40, 190],
        [0, 201, 161],
        [0, 201, 161],
        [0, 201, 161],
        [0, 201, 161],
        [60, 199, 0],
        [60, 199, 0],
        [60, 199, 0],
        [60, 199, 0],
        [158, 81, 34],
        [158, 81, 34],
        [158, 81, 34],
        [158, 81, 34],
        [205, 0, 179],
        [205, 0, 179],
        [205, 0, 179],
        [205, 0, 179],
    ]

    # limbSeq
    for i in range(len(limbSeq)):
        idx = limbSeq[i]
        x1, y1, c1 = json_data[idx[0]]
        x2, y2, c2 = json_data[idx[1]]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))  # ensure coordinates are int
            color = limSeq_color[i]
            thickness = 4
            cv2.line(
                canvas,
                start_point,
                end_point,
                matplotlib.colors.hsv_to_rgb([i / float(len(limbSeq)), 1.0, 1.0])
                * 255,  # rbg  brg
                thickness,
            )  # draw line
    # joint
    for i in range(len(json_data)):
        if i != 4:
            x, y, c = json_data[i]

            cv2.circle(canvas, (int(x), int(y)), 5, [0, 0, 255], thickness=-1)

    return canvas


def draw_hand_left(json_data, canvas):
    limbSeq = [
        [0, 1],
        [1, 2],
        [2, 3],
        # [3, 4],  # 엄지
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],  # 검지
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],  # 중지
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],  # 약지
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]

    # limbSeq
    for i in range(len(limbSeq)):
        idx = limbSeq[i]
        x1, y1, c1 = json_data[idx[0]]
        x2, y2, c2 = json_data[idx[1]]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))  # ensure coordinates are int
            thickness = 4
            cv2.line(
                canvas,
                start_point,
                end_point,
                matplotlib.colors.hsv_to_rgb(
                    [((len(limbSeq) - i - 3) / float(len(limbSeq))), 1.0, 1.0]
                )
                * 255,  # rbg  brg
                thickness,
            )  # draw line
    # joint
    for i in range(len(json_data)):
        if i != 4:
            x, y, c = json_data[i]

            cv2.circle(canvas, (int(x), int(y)), 5, [0, 0, 255], thickness=-1)

    return canvas


## 파일 불러오기
# json_path = f"./hand/03_real_word_keypoint/NIA_SL_WORD0006_REAL03_F/NIA_SL_WORD0006_REAL03_F_000000000{i:03d}_keypoints.json"
# Load json data
json_path = "./hand/hand/10_real_word_keypoint/NIA_SL_WORD0022_REAL10_F/NIA_SL_WORD0022_REAL10_F_000000000"
filename = []
for i in range(0, 132, 3):
    filename.append(json_path + f"{i:03d}_keypoints.json")
# print(filename)
# print(filename)

for name in filename:
    data_right = json2keypoints(name, "right")
    data_left = json2keypoints(name, "left")
    # data_left = np.flip(data_left, axis=0)

    # canvas 설정
    canvas_right = np.zeros((1280, 2000, 3), dtype=np.uint8)
    canvas_left = np.zeros((1280, 2000, 3), dtype=np.uint8)

    # Draw the face pose on the canvas
    canvas_right = draw_hand_right(data_right, canvas_right)
    canvas_left = draw_hand_left(data_left, canvas_left)

    cv2.imwrite(
        f"./res_hand_pose_image/pose_right{name[-18:-15]}_new.png", canvas_right
    )
    cv2.imwrite(f"./res_hand_pose_image/pose_left{name[-18:-15]}_new.png", canvas_left)

    # rgb
    rgb_image_right = cv2.cvtColor(canvas_right, cv2.COLOR_BGR2RGB)
    rgb_image_left = cv2.cvtColor(canvas_left, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        f"./controlnet/res_hand_pose_image/pose_right{name[-18:-15]}_rgb.png",
        rgb_image_right,
    )
    cv2.imwrite(
        f"./controlnet/res_hand_pose_image/pose_left{name[-18:-15]}_rgb.png",
        rgb_image_left,
    )
