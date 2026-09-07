# 脊柱统一推理引擎

这个工程将 `spine_detection_package` 和 `lumbar_spondylolisthesis_package` 的核心能力合并为一个无状态推理服务。

当前只保留三类能力：

- 影像输入
- 关键点推理
- 基于关键点的疾病指标计算

当前已支持：

- JPG / PNG 等普通影像
- 未压缩 DICOM
- 常见 JPEG Lossless 压缩 DICOM
- 浏览器端无法直接解析的 DICOM 会自动走后端预览兜底

当前不包含：

- 报告生成
- 中间结果落盘
- 会话状态缓存

## 目录

- `app/api.py`：Flask 接口入口
- `app/config.py`：模型路径、设备和任务配置
- `app/services/`：模型注册与任务流水线
- `app/algorithms/`：侧弯和滑脱指标计算
- `assets/`：统一工程自带的模型权重与模型结构代码
- `docs/model-assets.md`：模型命名、任务依赖和本地权重说明
- `run.py`：本地启动入口

## 接口

## 标准流程

- 第一步：上传医学影像，调用 `keypoints` 接口得到关键点
- 第二步：前端可选修改关键点位置
- 第三步：提交最终确认的关键点，调用 `analysis` 接口得到指标
- `analyze` 接口仍保留，用于“上传影像后一步直出”的快捷测试

### 健康检查

- `GET /health`

### 测试界面

- `GET /`
- 浏览器中可直接上传影像并调用统一接口

### 影像预览

- `POST /api/preview`
- 用于将上传影像转成浏览器可显示的 PNG 预览

### Swagger 文档

- `GET /apidocs/`
- OpenAPI 规范地址：`GET /apispec_1.json`

### 侧弯关键点

- `POST /api/scoliosis/keypoints`
- 表单文件字段：`ap_image`、`lat_image`

### 侧弯完整分析

- `POST /api/scoliosis/analyze`
- 表单文件字段：`ap_image`、`lat_image`

### 侧弯基于关键点分析

- `POST /api/scoliosis/analysis`
- JSON 字段：`views.ap.keypoints`、`views.lat.keypoints`

### 滑脱关键点

- `POST /api/slippage/keypoints`
- 表单文件字段：`lat_image`、`gs_image`、`gq_image`
- 侧位角计算放宽规则：
  - TK：至少检测到 12 节椎骨时，按 `T1 ~ T12` 计算
  - LL：至少检测到 14 节椎骨时，按 `L1 ~ 最后一节可用腰椎` 近似计算

### 滑脱完整分析

- `POST /api/slippage/analyze`
- 表单文件字段：`lat_image`、`gs_image`、`gq_image`

### 滑脱基于关键点分析

- `POST /api/slippage/analysis`
- JSON 字段：`views.lat.keypoints`、`views.gs.keypoints`、`views.gq.keypoints`
- 指标说明：
  - TK 不再强制要求 17 节椎骨，满足 `T1 ~ T12` 可用即可输出
  - LL 不再强制要求完整 `L1 ~ L5`，满足 `L1` 到最后一节可用腰椎时可近似输出

## 环境准备

推荐使用项目自己的 `.venv`，不要复用旧工程虚拟环境。

### 一键创建 venv

```powershell
.\scripts\setup_venv.ps1
```

这个脚本会做 4 件事：

- 创建 `.venv`
- 将 pip 默认源配置为清华源
- 安装通用依赖
- 单独安装 GPU 版 PyTorch

### 手动创建 venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PIP_CONFIG_FILE = "$PWD\pip.tuna.ini"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -r requirements-gpu-cu124.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### CUDA 说明

- 当前机器驱动显示支持 CUDA 12.6
- 工程默认安装 `PyTorch 2.4.1 + cu124`
- 这是兼容方案，CUDA 12.6 驱动可以向下兼容运行 cu124 轮子
- 如果后面验证稳定，再考虑整体升级到 `torch 2.10 + cu126`

### 验证 GPU

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## 启动

```bash
.\.venv\Scripts\Activate.ps1
python run.py
```

启动后可直接打开：

- `http://127.0.0.1:8001/`

## 环境变量

- `SPINE_DEVICE`：`auto`、`cpu`、`cuda`
- `SPINE_HOST`：默认 `0.0.0.0`
- `SPINE_PORT`：默认 `8001`
- `SPINE_WARMUP_MODELS`：`1` 时启动后预热所有模型

## 说明

当前工程已经把旧项目依赖的模型权重和模型结构代码迁移到自身目录：

- [assets](file:///g:/脊柱/spine_unified_package/assets)
- [scoliosis](file:///g:/脊柱/spine_unified_package/assets/scoliosis)
- [slippage](file:///g:/脊柱/spine_unified_package/assets/slippage)

其中：

- `assets/scoliosis/models` 保存侧弯相关权重
- `assets/scoliosis/model_architecture` 保存侧弯关键点模型结构
- `assets/slippage/models` 保存滑脱相关权重和 HRNet 预训练权重
- `assets/slippage/model_architecture` 保存滑脱关键点模型结构
