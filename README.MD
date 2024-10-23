# Prerequisite

[![Generic badge](https://img.shields.io/badge/CUDA-11.4-brightgreen.svg)](https://developer.nvidia.com/cuda-11-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local)
![Generic badge](https://img.shields.io/badge/pytorch-1.12.1-orange.svg)
![Generic badge](https://img.shields.io/badge/mmcv_full-1.6.2-red.svg)
![Generic badge](https://img.shields.io/badge/mmdet-1.12.1-red.svg)
![Generic badge](https://img.shields.io/badge/mmpose-1.12.1-red.svg)

This repository is based on [anime-face-detector](https://github.com/hysts/anime-face-detector)

![img.png](assets/img.png)
![img_1.png](assets/img_1.png)

<br/>

## Installation

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet==2.28.2
pip install mmpose==0.29.0
pip install numpy==1.23.5
```

This package is tested on Windows.
