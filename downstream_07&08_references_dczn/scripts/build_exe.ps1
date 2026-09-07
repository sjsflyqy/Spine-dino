# build_exe.ps1 —— 一键打包脊柱统一推理引擎为 exe
# 用法: powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
#
# 前提条件：已激活虚拟环境并安装好所有依赖
# 产出: dist\spine_unified\ 目录，可整体复制到目标机器部署

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SpecFile = Join-Path $ProjectRoot "spine_unified.spec"
$DistDir = Join-Path $ProjectRoot "dist" "spine_unified"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " 脊柱统一推理引擎 —— 打包 EXE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查 pyinstaller 是否可用
$pi = Get-Command pyinstaller -ErrorAction SilentlyContinue
if (-not $pi) {
    Write-Host "正在安装 PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
}

# 2. 执行 PyInstaller 打包
Write-Host ""
Write-Host "[1/3] 正在执行 PyInstaller 打包..." -ForegroundColor Green
Push-Location $ProjectRoot
pyinstaller --noconfirm $SpecFile
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Write-Host "打包失败！" -ForegroundColor Red
    exit 1
}

# 3. 复制模型权重文件到输出目录（不打进 exe，方便后续更新模型）
Write-Host ""
Write-Host "[2/3] 正在复制模型文件到输出目录..." -ForegroundColor Green

$AssetsSource = Join-Path $ProjectRoot "assets"
$AssetsDest = Join-Path $DistDir "assets"

# 复制 scoliosis 模型
$ScoliosisModelDest = Join-Path $AssetsDest "scoliosis" "models"
New-Item -ItemType Directory -Path $ScoliosisModelDest -Force | Out-Null
Copy-Item (Join-Path $AssetsSource "scoliosis" "models" "*.pt") $ScoliosisModelDest -Force
Copy-Item (Join-Path $AssetsSource "scoliosis" "models" "*.pth") $ScoliosisModelDest -Force

# 复制 slippage 模型
$SlippageModelDest = Join-Path $AssetsDest "slippage" "models"
New-Item -ItemType Directory -Path $SlippageModelDest -Force | Out-Null
Copy-Item (Join-Path $AssetsSource "slippage" "models" "*.pt") $SlippageModelDest -Force
Copy-Item (Join-Path $AssetsSource "slippage" "models" "*.pth") $SlippageModelDest -Force

# 4. 复制启动脚本
Write-Host ""
Write-Host "[3/3] 正在创建启动脚本..." -ForegroundColor Green

$StartBat = Join-Path $DistDir "start.bat"
if (-not (Test-Path $StartBat)) {
    Copy-Item (Join-Path $ProjectRoot "start.bat") $DistDir -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " 打包完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "输出目录: $DistDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "部署步骤:" -ForegroundColor Yellow
Write-Host "  1. 将 dist\spine_unified\ 整个文件夹复制到目标电脑"
Write-Host "  2. 双击 start.bat 启动服务"
Write-Host "  3. 局域网访问 https://<目标机器IP>:8001"
Write-Host ""
Write-Host "目录结构:" -ForegroundColor Yellow
Write-Host "  spine_unified\"
Write-Host "    spine_unified.exe    -- 主程序"
Write-Host "    start.bat            -- 启动脚本"
Write-Host "    assets\              -- 模型权重（可单独更新）"
Write-Host "    certs\               -- HTTPS 证书（首次启动自动生成）"
Write-Host "    _internal\           -- 运行时依赖（不要删除）"
