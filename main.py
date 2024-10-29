import os
import warnings
import pandas as pd

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

def save_results_to_csv(data, filename):
    """결과를 CSV 파일로 저장하는 함수"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    original_image_folder = './sample/yokmang_arin/original_images'
    sample_image_folder = './sample/yokmang_arin/22700'
    csv_filename = os.path.relpath(sample_image_folder, './sample') + '.csv'

    output_folder = './outputs'
    os.makedirs(os.path.join(output_folder, os.path.dirname(csv_filename)), exist_ok=True)
    csv_filepath = os.path.join(output_folder, csv_filename)

    original_body_ratios, original_std_ratios, original_min_ratios, original_max_ratios, original_valid_images = analyze_body_folder(original_image_folder)
    sample_body_ratios, sample_std_ratios, sample_min_ratios, sample_max_ratios, sample_valid_images = analyze_body_folder(sample_image_folder)

    combined_results = []

    if original_body_ratios is not None and sample_body_ratios is not None:
        for key in original_body_ratios.keys():
            sample_value = sample_body_ratios[key]
            original_value = original_body_ratios[key]
            difference = calculate_error_rate(sample_value, original_value)
            combined_results.append({
                "Type": "Body",
                "Ratio": key.replace('_', ' ').title(),
                "Sample Avg": sample_value,
                "Original Avg": original_value,
                "Difference (%)": difference,
                "Original Std Dev": original_std_ratios[key],
                "Sample Std Dev": sample_std_ratios[key],
                "Original Min": original_min_ratios[key],
                "Sample Min": sample_min_ratios[key],
                "Original Max": original_max_ratios[key],
                "Sample Max": sample_max_ratios[key]
            })
            print(f"Body {key.replace('_', ' ').title()} Ratio: Sample Avg={sample_value:.4f}, Original Avg={original_value:.4f}, Difference={difference:.2f}%")
            print(f"  Original Std Dev={original_std_ratios[key]:.4f}, Sample Std Dev={sample_std_ratios[key]:.4f}")
            print(f"  Original Min={original_min_ratios[key]:.4f}, Sample Min={sample_min_ratios[key]:.4f}")
            print(f"  Original Max={original_max_ratios[key]:.4f}, Sample Max={sample_max_ratios[key]:.4f}")
    else:
        print("One of the body folders did not return valid analysis results.")

    original_images = load_images_from_folder(original_image_folder)
    sample_images = load_images_from_folder(sample_image_folder)

    original_average_ratios, original_std_ratios, original_min_ratios, original_max_ratios, original_valid_images = calculate_average_and_std_ratios(original_images)
    if original_average_ratios is None:
        print("원본 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        print(f"원본 이미지 비율 계산 완료 (탐지된 이미지 수: {original_valid_images}/{len(original_images)})")

    sample_average_ratios, sample_std_ratios, sample_min_ratios, sample_max_ratios, sample_valid_images = calculate_average_and_std_ratios(sample_images)
    if sample_average_ratios is None:
        print("샘플 이미지에서 얼굴을 찾지 못했습니다.")
    else:
        print(f"샘플 이미지 비율 계산 완료 (탐지된 이미지 수: {sample_valid_images}/{len(sample_images)})")

    if original_average_ratios and sample_average_ratios:
        percentage_difference = calculate_percentage_difference_with_values(original_average_ratios, sample_average_ratios)
        print("\n비율 차이 (원본 대비 추론 값 및 차이, 표준 편차 포함):")
        for name, (original_value, sample_value, diff) in percentage_difference.items():
            combined_results.append({
                "Type": "Face",
                "Ratio": name.replace('_', ' ').title(),
                "Original Avg": original_value,
                "Sample Avg": sample_value,
                "Difference (%)": diff,
                "Original Std Dev": original_std_ratios[name],
                "Sample Std Dev": sample_std_ratios[name],
                "Original Min": original_min_ratios[name],
                "Sample Min": sample_min_ratios[name],
                "Original Max": original_max_ratios[name],
                "Sample Max": sample_max_ratios[name]
            })
            print(f'{name}:')
            print(f'  원본 평균: {original_value:.4f}, 표준 편차: {original_std_ratios[name]:.4f}')
            print(f'  추론 평균: {sample_value:.4f}, 표준 편차: {sample_std_ratios[name]:.4f}')
            print(f'  차이: {diff:.2f}%\n')

    save_results_to_csv(combined_results, csv_filepath)

if __name__ == "__main__":
    main()
