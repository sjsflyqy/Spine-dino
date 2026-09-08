from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import cv2
import numpy as np


def _decode_dicom(raw: bytes) -> np.ndarray:
    # 这个函数负责把原始 DICOM 二进制内容转换为 OpenCV 可继续处理的 BGR 图像。
    # 统一放在后端做，是为了兼容浏览器前端难以稳定支持的压缩和特殊编码 DICOM。
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
    except ModuleNotFoundError as exc:
        raise ValueError("当前环境未安装 pydicom，无法解析 DICOM 影像") from exc

    dataset = pydicom.dcmread(BytesIO(raw), force=True)
    if not hasattr(dataset, "PixelData"):
        raise ValueError("DICOM 文件缺少像素数据")

    # pixel_array 负责把 DICOM 中的像素数据真实解码出来。
    # apply_modality_lut / apply_voi_lut 则尽量把医学灰度映射到更适合显示的范围。
    pixels = dataset.pixel_array
    pixels = apply_modality_lut(pixels, dataset)
    try:
        pixels = apply_voi_lut(pixels, dataset)
    except Exception:
        # 某些 DICOM 没有 VOI LUT 或者 LUT 信息不完整，这里允许直接跳过。
        pass

    pixels = np.asarray(pixels).astype(np.float32)
    if pixels.ndim == 3 and pixels.shape[-1] == 3:
        # 多通道 DICOM 直接沿用原始通道数据。
        image = pixels
    else:
        # MONOCHROME1 需要先反相，否则骨骼会黑白颠倒。
        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = pixels.max() - pixels

        # 统一拉伸到 0~255，便于后续前端预览和模型推理复用同一图像。
        pixels -= pixels.min()
        max_value = float(pixels.max())
        if max_value > 0:
            pixels = pixels / max_value
        image = (pixels * 255.0).clip(0, 255).astype(np.uint8)
        # 关键点和检测模型都按三通道图像处理，因此灰度图也转成 BGR。
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image.astype(np.uint8)


def decode_image(file_storage: BinaryIO) -> np.ndarray:
    # 统一影像解码入口：
    # 先尝试当普通图片解析，失败后再按 DICOM 解析。
    raw = file_storage.read()
    if not raw:
        raise ValueError("空文件无法解析")
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        return _decode_dicom(raw)
    except Exception as exc:
        raise ValueError(f"无法解析影像文件: {exc}") from exc


def render_preview_png(file_storage: BinaryIO, max_side: int = 1600) -> bytes:
    # 这个函数专门给前端预览接口使用。
    # 目标不是保留医学原始像素，而是生成一个浏览器一定能显示的 PNG 缩略图。
    image = decode_image(file_storage)
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side > max_side:
        # 对超大影像做缩放，避免一次返回过大的 PNG 影响页面响应速度。
        scale = max_side / float(longest_side)
        image = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("预览图编码失败")
    return encoded.tobytes()
