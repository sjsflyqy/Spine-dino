# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec 文件 —— 脊柱统一推理引擎
# 用法: pyinstaller spine_unified.spec
#
# 打包策略：
#   - 代码和轻量资源（模板、静态文件、flasgger UI）打进 exe 内部
#   - 模型权重和 assets 目录放在 exe 同级，不打进 exe（避免 exe 过大、方便更新模型）
#   - 证书目录 certs/ 也在 exe 同级，首次 HTTPS 启动时自动生成

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_root = Path(SPECPATH)

# ── 需要打进 exe 的数据文件 ──────────────────────────────────
datas = [
    # Flask 模板和静态文件
    (str(project_root / 'app' / 'templates'), 'app/templates'),
    (str(project_root / 'app' / 'static'), 'app/static'),
]

# flasgger 自带的 Swagger UI 资源
datas += collect_data_files('flasgger')

# ultralytics 运行时需要的 cfg/default.yaml 等
datas += collect_data_files('ultralytics')

# timm 模型注册表
datas += collect_data_files('timm', include_py_files=False)

# ── 隐式导入 ─────────────────────────────────────────────────
hiddenimports = [
    # 项目内部模块
    'app',
    'app.api',
    'app.config',
    'app.schemas',
    'app.services',
    'app.services.model_registry',
    'app.services.scoliosis',
    'app.services.slippage',
    'app.algorithms',
    'app.algorithms.cobb',
    'app.algorithms.slippage',
    'app.utils',
    'app.utils.image',
    # 模型结构（assets 下的 Python 代码在 exe 内）
    'assets',
    'assets.scoliosis',
    'assets.scoliosis.model_architecture',
    'assets.scoliosis.model_architecture.scoliosis_spinenet',
    'assets.slippage',
    'assets.slippage.model_architecture',
    'assets.slippage.model_architecture.hrnet18_backbone',
    'assets.slippage.model_architecture.hrnet18_config',
    'assets.slippage.model_architecture.slippage_spinenet',
    # 第三方库隐式依赖
    'cryptography',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.primitives.asymmetric',
    'cryptography.hazmat.primitives.asymmetric.rsa',
    'cryptography.hazmat.primitives.hashes',
    'cryptography.hazmat.primitives.serialization',
    'cryptography.x509',
    'engineio.async_drivers.threading',
    'PIL',
    'pydicom',
    'pydicom.encoders',
    'pydicom.encoders.gdcm',
    'pydicom.encoders.pylibjpeg',
    'pylibjpeg',
    'pylibjpeg.plugins',
    'dill',
    'torch',
    'torchvision',
    'flask_cors',
    'flasgger',
    'yacs',
    'yacs.config',
]

# 收集 ultralytics 全部子模块，避免动态导入遗漏
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('timm')

# ── 模型结构 Python 代码也要打进 exe（作为 data）──────────────
# 这些 .py 文件会被 torch.load / importlib 动态加载
datas += [
    (str(project_root / 'assets' / 'scoliosis' / 'model_architecture'), 'assets/scoliosis/model_architecture'),
    (str(project_root / 'assets' / 'slippage' / 'model_architecture'), 'assets/slippage/model_architecture'),
    (str(project_root / 'assets' / '__init__.py'), 'assets'),
    (str(project_root / 'assets' / 'scoliosis' / '__init__.py'), 'assets/scoliosis'),
    (str(project_root / 'assets' / 'slippage' / '__init__.py'), 'assets/slippage'),
]

a = Analysis(
    [str(project_root / 'run.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tkinter', 'IPython', 'notebook', 'jupyter',
        'pytest', 'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='spine_unified',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,       # 不使用 UPX 压缩，避免 torch DLL 被损坏
    console=True,     # 控制台窗口，方便查看日志
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='spine_unified',
)
