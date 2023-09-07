import numpy as np
import json
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


## jason to keypoint ##
def json2keypoints(json_file, points):
    with open(json_file, "r") as file:
        data = json.load(file)
    # data_json = data['people']['face_keypoints_2d']
    keypoints = np.array(data["people"][points]).reshape(-1, 3)
    return keypoints


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


## draw face, hand, body ##


def draw_face(keypoints, canvas):
    for i in range(len(keypoints)):
        x, y, c = keypoints[i]
        # print(int(x), int(y))
        cv2.circle(canvas, (int(x), int(y)), 4, [225, 225, 225], thickness=-1)
    return canvas


def draw_body(json_data, canvas):
    stickwidth = 5
    limbSeq = [
        [2, 3],
        [2, 6],  # 어깨
        [3, 4],
        [4, 5],  # 오른팔
        [6, 7],
        [7, 8],  # 왼팔
        [2, 10],  # right hip
        [2, 13],  # left hip
        [2, 1],  # 목
        [1, 16],
        [16, 18],  # 오른 얼굴
        [1, 17],
        [17, 19],  # 왼 얼굴
    ]

    colors = [
        [153, 0, 1],
        [153, 51, 1],  # 어깨
        [152, 102, 1],
        [153, 153, 0],  # 오른팔
        [102, 152, 0],
        [55, 152, 9],  # 왼팔
        [4, 154, 1],  # left
        [28, 148, 145],  # right
        [0, 0, 152],  # 목
        [59, 1, 151],
        [99, 1, 153],  # 오른 얼굴
        [193, 5, 188],
        [160, 0, 104],  # 왼 얼굴
    ]

    joint_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [155, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 0, 0],
        [4, 154, 1],
        [0, 0, 0],  # 10
        [0, 0, 0],
        [28, 148, 145],
        [0, 0, 0],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 0, 85],  # 18
    ]

    # Draw limbs on the canvas
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        x1, y1, c1 = json_data[index[0] - 1]
        x2, y2, c2 = json_data[index[1] - 1]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))  # ensure coordinates are int
            color = colors[i]
            thickness = 10
            cv2.line(canvas, start_point, end_point, colors[i], thickness)  # draw line
    # Draw keypoints on the canvas

    num = list(range(8)) + list(range(15, 20)) + [9, 12]
    for i in range(19):
        if i in num:
            x, y, c = json_data[i]
            if c != 0:
                cv2.circle(canvas, (int(x), int(y)), 4, joint_colors[i], thickness=10)

    return canvas


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


## 파일 불러오기 ##
json_path = "./hand/hand/10_real_word_keypoint/NIA_SL_WORD0022_REAL10_F/NIA_SL_WORD0022_REAL10_F_000000000"
filename = []
for i in range(0, 132, 3):
    filename.append(json_path + f"{i:03d}_keypoints.json")


def json2openpose(file, crop_path):
    for idx, file in enumerate(filename):
        data_face = json2keypoints(file, "face_keypoints_2d")
        data_body = json2keypoints(file, "pose_keypoints_2d")
        data_hand_left = json2keypoints(file, "pose_left_2d")
        data_hand_right = json2keypoints(file, "hand_right_keypoints_2d")
        face = np.np.zeros((1280, 2000, 3), dtype=np.uint8)
        body = np.zeros((1280, 2000, 3), dtype=np.uint8)
        left_hand = np.zeros((1280, 2000, 3), dtype=np.uint8)
        right_hand = np.zeros((1280, 2000, 3), dtype=np.uint8)

        face = draw_face(data_face, face)
        body = draw_body(data_body, body)
        left_hand = draw_body(data_hand_left, left_hand)
        right_hand = draw_body(data_hand_right, right_hand)

        alpha = 1
        beta = 1
        gamma = 1
        composite_hand = cv2.addWeighted(left_hand, alpha, right_hand, beta, 0)
        composite_hand_body = cv2.addWeighted(composite_hand, alpha, body, beta, 0)
        composite_all = cv2.addWeighted(composite_hand_body, alpha, face, beta, 0)

        resized_img = resize_image(composite_all, 80)
        crop_width = 540
        crop_height = 960
        crop_path = (
            f"./controlnet/res_total_pose/pose_total_crop_960x540_{(idx)*3:03d}_new.png"
        )

        img_height, img_width, _ = resized_img.shape

        start_x = (img_width - crop_width - 60) // 2
        start_y = (img_height - crop_height - 60) // 2
        end_x = start_x + crop_width
        end_y = start_y + crop_height
        cropped_image = resized_img[start_y:end_y, start_x:end_x]
        print("크롭 이미지 저장", crop_path)
        cv2.imwrite(crop_path, cropped_image)
