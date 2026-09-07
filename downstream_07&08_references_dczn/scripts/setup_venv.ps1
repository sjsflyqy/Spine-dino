$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = "C:\Users\TANG\AppData\Local\Programs\Python\Python310\python.exe"
$PipConfigPath = Join-Path $ProjectRoot "pip.tuna.ini"

if (-not (Test-Path $PythonExe)) {
    throw "未找到 Python 3.10，可修改脚本中的 PythonExe 后重试"
}

if (-not (Test-Path $VenvPath)) {
    & $PythonExe -m venv $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"

$env:PIP_CONFIG_FILE = $PipConfigPath

& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPip install `
    -r (Join-Path $ProjectRoot "requirements.txt") `
    -r (Join-Path $ProjectRoot "requirements-gpu-cu124.txt") `
    --extra-index-url https://download.pytorch.org/whl/cu124

Write-Host ""
Write-Host "虚拟环境已创建完成：" $VenvPath
Write-Host "激活命令："
Write-Host "  .\.venv\Scripts\Activate.ps1"
