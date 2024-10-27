import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
current_directory = os.path.dirname(os.path.abspath(__file__))

# 원본 이미지 폴더와 샘플 이미지 폴더 경로
original_image_folder = "./sample/original_images"
sample_image_folder = "./sample/inference_images"
BG_COLOR = (192, 192, 192)  # 회색

def calculate_normalized_distance(landmark1, landmark2):
    """정규화된 유클리드 거리를 계산하는 함수"""
    x1, y1, z1 = landmark1.x, landmark1.y, landmark1.z
    x2, y2, z2 = landmark2.x, landmark2.y, landmark2.z
    return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def calculate_ratio(value1, value2):
    """두 값의 비율을 계산하는 함수"""
    if value2 == 0:
        return 0
    return value1 / value2

def calculate_error_rate(sample_value, original_value):
    """오차율을 계산하는 함수"""
    return abs((sample_value - original_value) / original_value) * 100

def analyze_image(image_path):
    """이미지에서 신체 비율을 분석하는 함수"""
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.1) as pose:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return None

        image_height, image_width, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print(f"No pose landmarks found in image: {image_path}")
            return None

        # 각 랜드마크 간의 거리 계산
        landmarks = results.pose_landmarks.landmark

        # 오른쪽 팔과 다리
        shoulder_width = calculate_normalized_distance(landmarks[11], landmarks[12])
        waist_width = calculate_normalized_distance(landmarks[23], landmarks[24])
        right_upper_arm = calculate_normalized_distance(landmarks[12], landmarks[14])
        right_forearm = calculate_normalized_distance(landmarks[14], landmarks[16])
        right_thigh = calculate_normalized_distance(landmarks[24], landmarks[26])
        right_calf = calculate_normalized_distance(landmarks[26], landmarks[28])

        # 왼쪽 팔과 다리
        left_upper_arm = calculate_normalized_distance(landmarks[13], landmarks[15])
        left_forearm = calculate_normalized_distance(landmarks[15], landmarks[17])
        left_thigh = calculate_normalized_distance(landmarks[25], landmarks[27])
        left_calf = calculate_normalized_distance(landmarks[27], landmarks[29])

        # 비율 계산
        shoulder_to_waist_ratio = calculate_ratio(shoulder_width, waist_width)
        upper_to_forearm_ratio = calculate_ratio(right_upper_arm + left_upper_arm, right_forearm + left_forearm)
        thigh_to_calf_ratio = calculate_ratio(right_thigh + left_thigh, right_calf + left_calf)
        total_arm = right_upper_arm + right_forearm + left_upper_arm + left_forearm
        total_leg = right_thigh + right_calf + left_thigh + left_calf
        arm_to_leg_ratio = calculate_ratio(total_arm, total_leg)
        waist_to_leg_ratio = calculate_ratio(waist_width, total_leg / 2)

        return {
            'shoulder_to_waist': shoulder_to_waist_ratio,
            'upper_to_forearm': upper_to_forearm_ratio,
            'thigh_to_calf': thigh_to_calf_ratio,
            'arm_to_leg': arm_to_leg_ratio,
            'waist_to_leg': waist_to_leg_ratio
        }

def analyze_folder(folder_path):
    """폴더 내의 모든 이미지를 분석하고 평균 비율 값을 계산"""
    total_ratios = {
        'shoulder_to_waist': 0,
        'upper_to_forearm': 0,
        'thigh_to_calf': 0,
        'arm_to_leg': 0,
        'waist_to_leg': 0
    }
    num_images = 0

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        ratios = analyze_image(image_path)
        if ratios is not None:
            for key in total_ratios.keys():
                total_ratios[key] += ratios[key]
            num_images += 1

    if num_images == 0:
        print(f"No valid images found in folder: {folder_path}")
        return None

    # 평균 계산
    average_ratios = {key: total_ratios[key] / num_images for key in total_ratios.keys()}
    return average_ratios

# 원본 이미지 폴더와 샘플 이미지 폴더 분석
original_average_ratios = analyze_folder(original_image_folder)
sample_average_ratios = analyze_folder(sample_image_folder)

if original_average_ratios and sample_average_ratios:
    # 평균 오차율 계산
    for key in original_average_ratios.keys():
        sample_value = sample_average_ratios[key]
        original_value = original_average_ratios[key]
        error_rate = calculate_error_rate(sample_value, original_value)
        print(f"{key.replace('_', ' ').title()} Ratio: Sample Avg={sample_value:.4f}, Original Avg={original_value:.4f}, Error={error_rate:.2f}%")
else:
    print("One of the folders did not return valid analysis results.")
