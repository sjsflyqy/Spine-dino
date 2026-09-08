from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch


# 统一工程现在已经把模型结构和权重都迁移到自身目录中，
# 后续部署时不再依赖旧工程目录是否存在。
import sys
_FROZEN = getattr(sys, "frozen", False)
if _FROZEN:
    # PyInstaller 打包后，_MEIPASS 是临时解压目录，
    # 但模型文件放在 exe 同级目录，所以用 exe 所在目录作为项目根。
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_ROOT = PROJECT_ROOT / "assets"
SCOLIOSIS_ROOT = ASSETS_ROOT / "scoliosis"
SLIPPAGE_ROOT = ASSETS_ROOT / "slippage"


def resolve_device() -> str:
    # 设备选择策略：
    # 1. 用户强制指定 cuda 时，只有显卡可用才真正返回 cuda
    # 2. 用户强制指定 cpu 时，始终走 cpu
    # 3. 默认 auto 模式下优先走 cuda，没有 GPU 再自动降级到 cpu
    requested = os.getenv("SPINE_DEVICE", "auto").lower()
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class YoloModelConfig:
    # name 用于内部注册和日志区分，weights 指向具体权重文件。
    name: str
    weights: Path


@dataclass(frozen=True)
class KeypointModelConfig:
    # architecture 用来标识当前权重对应哪套旧模型结构，
    # 便于统一工程在运行时按疾病类型加载兼容代码。
    name: str
    weights: Path
    architecture: str


# 这组环境变量控制服务监听地址、端口以及是否在启动时预热全部模型。
DEVICE = resolve_device()
HOST = os.getenv("SPINE_HOST", "0.0.0.0")
PORT = int(os.getenv("SPINE_PORT", "8001"))
WARMUP_MODELS = os.getenv("SPINE_WARMUP_MODELS", "0") == "1"

# HTTPS / SSL 配置
# 设置 SPINE_SSL_CERT 和 SPINE_SSL_KEY 环境变量即可开启 HTTPS，
# 也可以设置 SPINE_SSL=adhoc 让 Flask 用 pyOpenSSL 自动生成临时证书。
SSL_CERT = os.getenv("SPINE_SSL_CERT", "")
SSL_KEY = os.getenv("SPINE_SSL_KEY", "")
SSL_ADHOC = os.getenv("SPINE_SSL", "").lower() == "adhoc"


def _ensure_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    """如果证书文件不存在，则用 Python 标准库自动生成自签名证书。"""
    if cert_path.exists() and key_path.exists():
        return
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime, ipaddress, socket

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # 收集本机 IP 用于 SAN
        san_entries = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = ipaddress.IPv4Address(info[4][0])
                entry = x509.IPAddress(ip)
                if entry not in san_entries:
                    san_entries.append(entry)
        except Exception:
            pass

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "SpineUnified"),
        ])
        now = datetime.datetime.utcnow()
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
            .sign(key, hashes.SHA256())
        )

        cert_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        print(f"已自动生成自签名证书: {cert_path}, {key_path}")
    except ImportError:
        raise RuntimeError(
            "需要 cryptography 库来自动生成证书，请执行: pip install cryptography\n"
            "或者手动提供证书文件。"
        )


def get_ssl_context():
    """根据环境变量返回 Flask ssl_context 参数，不启用时返回 None。"""
    if SSL_CERT and SSL_KEY:
        cert = Path(SSL_CERT)
        key = Path(SSL_KEY)
        if not cert.is_absolute():
            cert = PROJECT_ROOT / cert
        if not key.is_absolute():
            key = PROJECT_ROOT / key
        # 文件不存在时自动生成
        _ensure_self_signed_cert(cert, key)
        return (str(cert), str(key))
    if SSL_ADHOC:
        return "adhoc"
    return None

# 统一把不同任务使用的视图、椎体标签和节段标签放在配置层，
# 避免在路由、前端和算法里再次硬编码。
SCOLIOSIS_VIEWS = ("ap", "lat")
SLIPPAGE_VIEWS = ("lat", "gs", "gq")
SLIPPAGE_LEVELS = ("L1", "L2", "L3", "L4", "L5", "S1")
VERTEBRA_LABELS = tuple([f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)])

# 侧弯任务使用 AP/LAT 两套椎体检测模型。
SCOLIOSIS_YOLO_MODELS = {
    "ap": YoloModelConfig("scoliosis_detector_ap_yolov8", SCOLIOSIS_ROOT / "models" / "scoliosis_detector_ap_yolov8.pt"),
    "lat": YoloModelConfig("scoliosis_detector_lat_yolov8", SCOLIOSIS_ROOT / "models" / "scoliosis_detector_lat_yolov8.pt"),
}

# 侧弯任务为不同视图分别使用一套关键点模型。
SCOLIOSIS_KEYPOINT_MODELS = {
    "ap": KeypointModelConfig("scoliosis_keypoints_ap_spinenet", SCOLIOSIS_ROOT / "models" / "scoliosis_keypoints_ap_spinenet.pth", "scoliosis"),
    "lat": KeypointModelConfig("scoliosis_keypoints_lat_spinenet", SCOLIOSIS_ROOT / "models" / "scoliosis_keypoints_lat_spinenet.pth", "scoliosis"),
}

# 滑脱任务只有一套腰椎区域检测模型和一套关键点模型。
SLIPPAGE_YOLO_MODEL = YoloModelConfig("slippage_detector_lumbar_yolov8", SLIPPAGE_ROOT / "models" / "slippage_detector_lumbar_yolov8.pt")
SLIPPAGE_KEYPOINT_MODEL = KeypointModelConfig("slippage_keypoints_spinenet", SLIPPAGE_ROOT / "models" / "slippage_keypoints_spinenet.pth", "slippage")
