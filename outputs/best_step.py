import os
import pandas as pd

# CSV 파일이 저장된 폴더 경로
csv_folder = './outputs/yokmang_arin'

# 각 비율에 대한 가중치
weights = {
    "Shoulder To Waist": 0.25,
    "Outer Canthal Distance": 0.1,
    "Inner Canthal Ratio": 0.1,
    "Eye To Nose Ratio": 0.15,
    "Nose To Mouth Ratio": 0.1,
    "Chin Length Ratio": 0.2,
    "Eyebrow Height Ratio": 0.1
}

def calculate_weighted_score(csv_file):
    """CSV 파일에서 가중합 점수를 계산하는 함수"""
    df = pd.read_csv(csv_file)
    score_details = {}
    total_score = 0.0
    for _, row in df.iterrows():
        ratio = row['Ratio']
        difference = row['Difference (%)']
        weight = weights.get(ratio, 0)
        weighted_score = difference * weight
        score_details[ratio] = weighted_score
        total_score += weighted_score
    return total_score, score_details

def find_best_csv(csv_folder):
    """가장 낮은 가중합 점수를 가진 CSV 파일을 찾는 함수"""
    best_score = float('inf')
    best_file = None

    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            csv_file = os.path.join(csv_folder, filename)
            total_score, score_details = calculate_weighted_score(csv_file)
            print(f"{filename}:")
            for ratio, weighted_score in score_details.items():
                print(f"  {ratio}: Weighted Score = {weighted_score:.4f}")
            print(f"  Total Weighted Score = {total_score:.4f}\n")
            if total_score < best_score:
                best_score = total_score
                best_file = filename

    return best_file, best_score

if __name__ == "__main__":
    best_file, best_score = find_best_csv(csv_folder)
    print(f"\nBest CSV File: {best_file} with Score: {best_score:.4f}")
