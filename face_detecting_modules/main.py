#@title import packages

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import anime_face_detector

#@title Contour Definition

# https://github.com/hysts/anime-face-detector/blob/main/assets/landmarks.jpg
FACE_BOTTOM_OUTLINE = np.arange(0, 5)
LEFT_EYEBROW = np.arange(5, 8)
RIGHT_EYEBROW = np.arange(8, 11)
LEFT_EYE_TOP = np.arange(11, 14)
LEFT_EYE_BOTTOM = np.arange(14, 17)
RIGHT_EYE_TOP = np.arange(17, 20)
RIGHT_EYE_BOTTOM = np.arange(20, 23)
NOSE = np.array([23])
MOUTH_OUTLINE = np.arange(24, 28)

FACE_OUTLINE_LIST = [FACE_BOTTOM_OUTLINE, LEFT_EYEBROW, RIGHT_EYEBROW]
LEFT_EYE_LIST = [LEFT_EYE_TOP, LEFT_EYE_BOTTOM]
RIGHT_EYE_LIST = [RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM]
NOSE_LIST = [NOSE]
MOUTH_OUTLINE_LIST = [MOUTH_OUTLINE]

# (indices, BGR color, is_closed)
CONTOURS = [
    (FACE_OUTLINE_LIST, (0, 170, 255), False),
    (LEFT_EYE_LIST, (50, 220, 255), False),
    (RIGHT_EYE_LIST, (50, 220, 255), False),
    (NOSE_LIST, (255, 30, 30), False),
    (MOUTH_OUTLINE_LIST, (255, 30, 30), True),
]

#@title Visualization Function


def visualize_box(image,
                  box,
                  score,
                  lt,
                  box_color=(0, 255, 0),
                  text_color=(255, 255, 255),
                  show_box_score=True):
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), box_color, lt)
    if not show_box_score:
        return
    cv2.putText(image,
                f'{round(score * 100, 2)}%', (box[0], box[1] - 2),
                0,
                lt / 2,
                text_color,
                thickness=max(lt, 1),
                lineType=cv2.LINE_AA)


def visualize_landmarks(image, pts, lt, landmark_score_threshold):
    for *pt, score in pts:
        pt = tuple(np.round(pt).astype(int))
        if score < landmark_score_threshold:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(image, pt, lt, color, cv2.FILLED)


def draw_polyline(image, pts, color, closed, lt, skip_contour_with_low_score,
                  score_threshold):
    if skip_contour_with_low_score and (pts[:, 2] < score_threshold).any():
        return
    pts = np.round(pts[:, :2]).astype(int)
    cv2.polylines(image, np.array([pts], dtype=np.int32), closed, color, lt)


def visualize_contour(image, pts, lt, skip_contour_with_low_score,
                      score_threshold):
    for indices_list, color, closed in CONTOURS:
        for indices in indices_list:
            draw_polyline(image, pts[indices], color, closed, lt,
                          skip_contour_with_low_score, score_threshold)


def visualize(image: np.ndarray,
              preds: np.ndarray,
              face_score_threshold: float,
              landmark_score_threshold: float,
              show_box_score: bool = True,
              draw_contour: bool = True,
              skip_contour_with_low_score=False):
    res = image.copy()

    for pred in preds:
        box = pred['bbox']
        box, score = box[:4], box[4]
        box = np.round(box).astype(int)
        pred_pts = pred['keypoints']

        # line_thickness
        lt = max(2, int(3 * (box[2:] - box[:2]).max() / 256))

        visualize_box(res, box, score, lt, show_box_score=show_box_score)
        if draw_contour:
            visualize_contour(
                res,
                pred_pts,
                lt,
                skip_contour_with_low_score=skip_contour_with_low_score,
                score_threshold=landmark_score_threshold)
        visualize_landmarks(res, pred_pts, lt, landmark_score_threshold)

    return res

#@title Detector

device = 'cuda:0'  #@param ['cuda:0', 'cpu']
model = 'yolov3'  #@param ['yolov3', 'faster-rcnn']
detector = anime_face_detector.create_detector(model, device=device)

#@title Visualization Arguments

face_score_threshold = 0.5  #@param {type: 'slider', min: 0, max: 1, step:0.1}
landmark_score_threshold = 0.3  #@param {type: 'slider', min: 0, max: 1, step:0.1}
show_box_score = True  #@param {'type': 'boolean'}
draw_contour = True  #@param {'type': 'boolean'}
skip_contour_with_low_score = True  #@param {'type': 'boolean'}

chin_index = 2  # 턱
nose_tip_index = 23  # 코 끝
left_mouth_index = 24  # 입의 왼쪽 끝
right_mouth_index = 26  # 입의 오른쪽 끝
mouth_center_index = 25  # 입의 중앙
left_brow_index = 6  # 왼쪽 눈썹 중앙
right_brow_index = 9  # 오른쪽 눈썹 중앙
eye_inner_left_index = 13  # 왼쪽 안쪽 눈
left_eye_index = 11  # 왼쪽 바깥 눈
eye_inner_right_index = 17  # 오른쪽 안쪽 눈
right_eye_index = 19  # 오른쪽 바깥 눈
face_left_index = 0  # 얼굴 광대 왼쪽
face_right_index = 4  # 얼굴 광대 오른쪽


# 거리 계산 함수
def calculate_distance(point1, point2):
    """두 점 사이의 유클리드 거리를 계산"""
    return np.linalg.norm(np.array(point1) - np.array(point2))


# 비율을 계산하는 함수
def calculate_ratios(landmarks):
    # 얼굴 너비와 높이 계산
    face_width = calculate_distance(landmarks[face_left_index], landmarks[face_right_index])

    # 얼굴 높이는 왼쪽과 오른쪽 눈썹 중간의 평균 지점과 턱 끝 사이 거리
    brow_center = (np.array(landmarks[left_brow_index]) + np.array(landmarks[right_brow_index])) / 2
    face_height = calculate_distance(brow_center, landmarks[chin_index])

    # 1. 눈 사이 거리 비율
    interocular_distance = calculate_distance(landmarks[left_eye_index], landmarks[right_eye_index])
    interocular_ratio = interocular_distance / face_width

    # 2. 눈-코 거리 비율
    eye_to_nose_distance = (calculate_distance(landmarks[left_eye_index], landmarks[nose_tip_index]) + calculate_distance(landmarks[right_eye_index], landmarks[nose_tip_index])) / 2
    eye_to_nose_ratio = eye_to_nose_distance / face_height

    # 3. 코-입 거리 비율
    nose_to_mouth_distance = calculate_distance(landmarks[nose_tip_index], landmarks[mouth_center_index])
    nose_to_mouth_ratio = nose_to_mouth_distance / face_height

    # 4. 입 너비 비율
    mouth_width = calculate_distance(landmarks[left_mouth_index], landmarks[right_mouth_index])
    mouth_width_ratio = mouth_width / face_width

    # 5. 턱 길이 비율 (턱 끝에서 코 끝까지)
    chin_to_nose_distance = calculate_distance(landmarks[chin_index], landmarks[nose_tip_index])
    chin_length_ratio = chin_to_nose_distance / face_height

    # 6. 눈썹 높이 비율 (눈과 눈썹 사이 거리)
    eyebrow_height = (calculate_distance(landmarks[left_eye_index], landmarks[left_brow_index]) + calculate_distance(landmarks[right_eye_index], landmarks[right_brow_index])) / 2
    eyebrow_height_ratio = eyebrow_height / face_height

    # 7. 미간 거리 비율 (눈 안쪽 모서리 사이 거리)
    intercanthal_distance = calculate_distance(landmarks[eye_inner_left_index], landmarks[eye_inner_right_index])
    intercanthal_ratio = intercanthal_distance / face_width

    return {
        'interocular_ratio': interocular_ratio,
        'eye_to_nose_ratio': eye_to_nose_ratio,
        'nose_to_mouth_ratio': nose_to_mouth_ratio,
        'mouth_width_ratio': mouth_width_ratio,
        'chin_length_ratio': chin_length_ratio,
        'eyebrow_height_ratio': eyebrow_height_ratio,
        'intercanthal_ratio': intercanthal_ratio
    }

def calculate_average_ratios(image_list, detector):
    total_ratios = {
        'interocular_ratio': 0,
        'eye_to_nose_ratio': 0,
        'nose_to_mouth_ratio': 0,
        'mouth_width_ratio': 0,
        'chin_length_ratio': 0,
        'eyebrow_height_ratio': 0,
        'intercanthal_ratio': 0
    }

    valid_images = 0  # 비율 계산이 성공한 이미지 수

    for image in image_list:
        preds = detector(image)  # 이미지에서 얼굴 랜드마크 추출
        if len(preds) > 0:
            keypoints = preds[0]['keypoints']
            ratios = calculate_ratios(keypoints)
            for key in total_ratios:
                total_ratios[key] += ratios[key]
            valid_images += 1

    if valid_images == 0:
        return None

    # 평균 구하기
    average_ratios = {key: value / valid_images for key, value in total_ratios.items()}
    return average_ratios

def calculate_percentage_difference(average1, average2):
    percentage_diff = {}
    for key in average1:
        # 원본 이미지와 샘플 이미지 비율의 차이를 백분율로 계산
        percentage_diff[key] = abs(average1[key] - average2[key]) / average1[key] * 100
    return percentage_diff

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

# image = cv2.imread('assets/KakaoTalk_20240716_001234703.png')
# preds = detector(image)

# print(preds)

original_images_folder = './original_images'
sample_images_folder = './sample_images'

original_images = load_images_from_folder(original_images_folder)
sample_images = load_images_from_folder(sample_images_folder)


# 원본 이미지의 평균 비율 계산
original_average_ratios = calculate_average_ratios(original_images, detector)
if original_average_ratios is None:
    print("원본 이미지에서 얼굴을 찾지 못했습니다.")
else:
    print("원본 이미지 비율 계산 완료")

# 샘플 이미지의 평균 비율 계산
sample_average_ratios = calculate_average_ratios(sample_images, detector)
if sample_average_ratios is None:
    print("샘플 이미지에서 얼굴을 찾지 못했습니다.")
else:
    print("샘플 이미지 비율 계산 완료")

# 두 집합 간의 비율 차이 계산 (백분율)
if original_average_ratios is not None and sample_average_ratios is not None:
    percentage_difference = calculate_percentage_difference(original_average_ratios, sample_average_ratios)

    # 결과 출력
    for name, diff in percentage_difference.items():
        print(f'{name}: {diff:.2f}% 차이')