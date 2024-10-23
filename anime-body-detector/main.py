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

# 이미지 파일의 경우 이것을 사용하세요.:
IMAGE_FILES = ["./3cd3e1e48a876425ede16db855b6ddd0.jpg"]
BG_COLOR = (192, 192, 192)  # 회색

print("Starting pose estimation...")  # 프로그램 시작 확인

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

with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.2) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        print(f"Processing file: {file}")  # 파일 처리 시작 확인
        image = cv2.imread(file)
        if image is None:
            print(f"Failed to read image: {file}")  # 이미지 파일 읽기 실패 시 출력
            continue

        # 처리 전 BGR 이미지를 RGB로 변환합니다.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            print("No pose landmarks found.")  # 포즈 랜드마크가 없는 경우 출력
            continue

        # 각 랜드마크 간의 거리 계산 (정규화된 좌표로 계산)
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
        # 1. 어깨 <-> 허리 비율
        shoulder_to_waist_ratio = calculate_ratio(shoulder_width, waist_width)
        
        # 2. (오른 윗팔 + 왼 윗팔의 평균) <-> (오른 아랫팔 + 왼 아랫팔의 평균)
        upper_to_forearm_ratio = calculate_ratio(right_upper_arm + left_upper_arm, right_forearm + left_forearm)

        # 3. (오른 윗다리 + 왼 윗다리의 평균) <-> (오른 아랫다리 + 왼 아랫다리의 평균)
        thigh_to_calf_ratio = calculate_ratio(right_thigh + left_thigh, right_calf + left_calf)

        # 4. 전체 팔 <-> 전체 다리 비율
        total_arm = right_upper_arm + right_forearm + left_upper_arm + left_forearm
        total_leg = right_thigh + right_calf + left_thigh + left_calf
        arm_to_leg_ratio = calculate_ratio(total_arm, total_leg)
        
        # 5. 허리 <-> 전체 다리 비율
        waist_to_leg_ratio = calculate_ratio(waist_width, total_leg/2)
        
        # 비율 출력
        print(f"Shoulder to Waist Ratio: {shoulder_to_waist_ratio:.4f}")
        print(f"Upper Arm to Forearm Ratio (average): {upper_to_forearm_ratio:.4f}")
        print(f"Thigh to Calf Ratio (average): {thigh_to_calf_ratio:.4f}")
        print(f"Total Arm to Total Leg Ratio: {arm_to_leg_ratio:.4f}")
        print(f"Waist to Total Leg Ratio: {waist_to_leg_ratio:.4f}")

        annotated_image = image.copy()
        # 이미지를 분할합니다.
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # 이미지 위에 포즈 랜드마크를 그립니다.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Matplotlib을 사용하여 플롯을 그리고 랜드마크 번호를 출력
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        
        # 각 랜드마크 번호를 이미지에 출력
        for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
            x = landmark.x * image.shape[1]
            y = landmark.y * image.shape[0]
            plt.text(x, y, str(landmark_id), color='red', fontsize=12, ha='center')

        plt.axis('off')  # 축을 숨깁니다.
        plt.show()  # 이미지를 표시합니다.

        # 결과 이미지를 파일로 저장
        output_path = os.path.join(current_directory, 'annotated_image' + str(idx) + '.png')
        print(f"Saving annotated image at: {output_path}")  # 파일 경로 출력
        success = cv2.imwrite(output_path, annotated_image)
        if success:
            print(f"Image saved successfully at: {output_path}")
        else:
            print(f"Failed to save image at: {output_path}")

print("Pose estimation complete.")
