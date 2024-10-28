import os
import warnings

# TensorFlow 로그 레벨 설정: 0 = 모든 로그 표시, 1 = 정보 로그 숨김, 2 = 경고 로그 숨김, 3 = 오류 로그 숨김
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 모든 경고 무시
warnings.filterwarnings("ignore")

from body_detecting_modules.body_detect import (
    calculate_average_and_std_ratios as analyze_body_folder,
    calculate_error_rate
)
from face_detecting_modules.face_detect import (
    load_images_from_folder,
    calculate_average_and_std_ratios,
    calculate_percentage_difference_with_values
)

def main():
    original_image_folder = './sample/elysia/original_images'
    sample_image_folder = './sample/elysia/inference_images/aria_elysia_v3-10475/full body'

    original_body_ratios, _ = analyze_body_folder(original_image_folder)
    sample_body_ratios, _ = analyze_body_folder(sample_image_folder)

    if original_body_ratios is not None and sample_body_ratios is not None:
        for key in original_body_ratios.keys():
            sample_value = sample_body_ratios[key]
            original_value = original_body_ratios[key]
            error_rate = calculate_error_rate(sample_value, original_value)
            print(f"Body {key.replace('_', ' ').title()} Ratio: Sample Avg={sample_value:.4f}, Original Avg={original_value:.4f}, Error={error_rate:.2f}%")
    else:
        print("One of the body folders did not return valid analysis results.")

    original_images = load_images_from_folder(original_image_folder)
    sample_images = load_images_from_folder(sample_image_folder)

    original_average_ratios, original_std_ratios, original_valid_images = calculate_average_and_std_ratios(original_images)
    if original_average_ratios is None:
        print("원본 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        print(f"원본 이미지 비율 계산 완료 (탐지된 이미지 수: {original_valid_images}/{len(original_images)})")

    sample_average_ratios, sample_std_ratios, sample_valid_images = calculate_average_and_std_ratios(sample_images)
    if sample_average_ratios is None:
        print("샘플 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        print(f"샘플 이미지 비율 계산 완료 (탐지된 이미지 수: {sample_valid_images}/{len(sample_images)})")

    if original_average_ratios and sample_average_ratios:
        percentage_difference = calculate_percentage_difference_with_values(original_average_ratios, sample_average_ratios)
        print("\n비율 차이 (원본 대비 추론 값 및 차이, 표준 편차 포함):")
        for name, (original_value, sample_value, diff) in percentage_difference.items():
            print(f'{name}:')
            print(f'  원본 평균: {original_value:.4f}, 표준 편차: {original_std_ratios[name]:.4f}')
            print(f'  추론 평균: {sample_value:.4f}, 표준 편차: {sample_std_ratios[name]:.4f}')
            print(f'  차이: {diff:.2f}%\n')

if __name__ == "__main__":
    main()
