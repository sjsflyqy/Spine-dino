@echo off
chcp 65001 >nul 2>&1
title 脊柱统一推理引擎

echo ========================================
echo  脊柱统一推理引擎
echo ========================================
echo.

:: 设置 HTTPS（证书不存在时会自动生成）
set SPINE_SSL_CERT=certs\server.crt
set SPINE_SSL_KEY=certs\server.key
set SPINE_PORT=8001

:: 如果目标机器有 NVIDIA GPU 且 CUDA 版本兼容，可注释下行以启用 GPU
:: 若出现 "CUDA error: no kernel image" 则必须保持 CPU 模式
set SPINE_DEVICE=cpu

echo 正在启动服务，请稍候...
echo 启动后请访问 https://本机IP:%SPINE_PORT%
echo 按 Ctrl+C 可停止服务
echo.

"%~dp0spine_unified.exe"

pause
