import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 읽기
file1 = 'outputs/aria_elysia_v3-850.csv'
file2 = 'outputs/aria_elysia_v3-10475.csv'

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# 차트를 위한 Figure 설정
plt.figure(figsize=(14, 7))
plt.title('Difference (%) with Sample Std Dev, Min, and Max for Each Ratio')
plt.xlabel('Ratio')
plt.ylabel('Difference (%)')

# 데이터 필터링 및 표준 편차, 최소값, 최대값 포함 차트 그리기
for i, (data, label) in enumerate(zip([data1, data2], ['File 1', 'File 2'])):
    ratios = data['Ratio']
    differences = data['Difference (%)']
    sample_std_dev = data['Sample Std Dev']
    sample_min = data['Sample Min']
    sample_max = data['Sample Max']
    
    # 차이값이 최소/최대 범위에서의 차이를 절대값으로 계산
    yerr = [abs(differences - sample_min), abs(sample_max - differences)]
    
    # x 위치를 약간 이동하여 겹침 방지
    x_positions = np.arange(len(ratios)) + (i * 0.1)

    # 상하방 표준 편차 및 최소/최대값 범위 포함하여 그리기
    plt.errorbar(x_positions, differences, yerr=[sample_std_dev, sample_std_dev], fmt='o', label=f'{label} Std Dev', capsize=5, alpha=0.7)
    plt.errorbar(x_positions, differences, yerr=yerr, fmt='o', alpha=0.3, capsize=3, label=f'{label} Min/Max Range')

plt.legend()
plt.xticks(np.arange(len(ratios)), ratios, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
