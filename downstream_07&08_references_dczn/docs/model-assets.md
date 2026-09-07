# 模型资产说明

## 目标

这个文档用于说明统一工程当前到底使用哪些模型、哪些模型属于哪个任务、哪些文件不能删除，以及当前是否还会从网络下载权重。

## 目录结构

- `assets/scoliosis/models`
  - `scoliosis_detector_ap_yolov8.pt`
  - `scoliosis_detector_lat_yolov8.pt`
  - `scoliosis_keypoints_ap_spinenet.pth`
  - `scoliosis_keypoints_lat_spinenet.pth`
- `assets/scoliosis/model_architecture`
  - `scoliosis_spinenet.py`
- `assets/slippage/models`
  - `slippage_detector_lumbar_yolov8.pt`
  - `slippage_keypoints_spinenet.pth`
  - `slippage_backbone_hrnet18_imagenet.pth`
- `assets/slippage/model_architecture`
  - `slippage_spinenet.py`
  - `hrnet18_backbone.py`
  - `hrnet18_config.py`

## 任务和模型对应关系

### 侧弯任务

- AP 椎体检测：`scoliosis_detector_ap_yolov8.pt`
- LAT 椎体检测：`scoliosis_detector_lat_yolov8.pt`
- AP 关键点：`scoliosis_keypoints_ap_spinenet.pth`
- LAT 关键点：`scoliosis_keypoints_lat_spinenet.pth`
- 结构代码：`scoliosis_spinenet.py`

### 滑脱任务

- 腰椎区域检测：`slippage_detector_lumbar_yolov8.pt`
- 滑脱关键点：`slippage_keypoints_spinenet.pth`
- HRNet18 主干预训练权重：`slippage_backbone_hrnet18_imagenet.pth`
- 结构代码：
  - `slippage_spinenet.py`
  - `hrnet18_backbone.py`
  - `hrnet18_config.py`

## 关于“滑脱没用到的模型”

滑脱任务**没有用到**下面两个模型：

- `scoliosis_detector_ap_yolov8.pt`
- `scoliosis_detector_lat_yolov8.pt`

但这两个文件仍然**不能删除**，因为它们属于侧弯任务的必需模型，不是滑脱目录里的冗余文件。

换句话说：

- 对滑脱任务来说，这两个模型无关
- 对整个统一工程来说，这两个模型仍然在被侧弯任务使用

## 当前是否还会联网下载权重

不会。

当前统一工程的模型相关代码已经改成：

- 侧弯主干不再走在线 ImageNet 下载
- 滑脱 HRNet 主干也不再走在线下载
- 所有运行所需权重都从 `assets/.../models` 本地目录加载

## 删除原则

当前这批文件中：

- `assets/slippage/models` 下 3 个文件都在实际使用，不能删
- `assets/scoliosis/models` 下 4 个文件都在实际使用，不能删
- 如果未来删除某个任务，才能整体删除该任务对应的一组模型和结构代码
