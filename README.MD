# LoRA evaluator

[![Generic badge](https://img.shields.io/badge/CUDA-11.4-brightgreen.svg)](https://developer.nvidia.com/cuda-11-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
[![Generic badge](https://img.shields.io/badge/pytorch-1.12.1-orange.svg)](https://pytorch.org/get-started/previous-versions/)
[![Generic badge](https://img.shields.io/badge/mmcv_full-1.6.2-red.svg)](https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html)
[![Generic badge](https://img.shields.io/badge/mmdet-1.12.1-red.svg)](https://github.com/open-mmlab/mmdetection)
[![Generic badge](https://img.shields.io/badge/mmpose-1.12.1-red.svg)](https://github.com/open-mmlab/mmpose)

 Lora-Evaluator is a powerful tool designed to combine anime face detection and body detection to analyze how similar a character is to its original form after model training. This repository integrates state-of-the-art technologies to assess the accuracy and consistency between the original and trained character models.

 
This repository is based on [anime-face-detector](https://github.com/hysts/anime-face-detector) repository and provides both face and body analysis to ensure precise similarity detection.

<img src="assets/img.png" width="400" height="300">
<img src="assets/img_1.png" width="300" height="300">


## Evaluate function

This library enables visual analysis to determine which LoRA step achieved better training results.

The evaluation function is defined as follows:

$$
\sum_{i=1}^{n} \left(| \text{avg(origin ratio)} - \text{avg(infer ratio)} \right| \times w_i) + \sum_{j=1}^{m} \left(| \text{avg(origin ratio)} - \text{avg(infer ratio)} \right| \times w_j)
$$

$n$ represents the number of detection metrics for the face, and $m$ represents the number of detection metrics for the body. $w_n$ and $w_m$ are weights selected based on the optimal balance as perceived by the user. 

These weights are derived through reverse calculation under the assumption of a linear relationship; however, nonlinear adjustments can be applied as needed.

Among the various LoRA steps, we will consider the step with the minimum value of this formula as the most optimal LoRA.

<br/>

## Installation
To get started with the tool, install the necessary libraries with the following commands:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet==2.28.2
pip install mmpose==0.29.0
pip install numpy==1.23.5
pip install mediapipe
```

⚙️ This package has been tested on Windows.

