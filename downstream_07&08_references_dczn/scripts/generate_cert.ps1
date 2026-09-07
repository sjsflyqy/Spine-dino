# generate_cert.ps1 —— 为局域网 HTTPS 访问生成自签名证书
# 用法: powershell -ExecutionPolicy Bypass -File scripts\generate_cert.ps1
#
# 生成的文件会放在项目根目录的 certs\ 文件夹下：
#   certs\server.key  —— 私钥
#   certs\server.crt  —— 证书（有效期 365 天）

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CertDir = Join-Path $ProjectRoot "certs"

if (-not (Test-Path $CertDir)) {
    New-Item -ItemType Directory -Path $CertDir | Out-Null
}

$KeyFile = Join-Path $CertDir "server.key"
$CrtFile = Join-Path $CertDir "server.crt"

# 获取本机所有 IPv4 地址，写入 SAN 以便局域网 IP 访问时证书也能匹配
$IPs = (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -ne "127.0.0.1" -and $_.PrefixOrigin -ne "WellKnown" } |
        Select-Object -ExpandProperty IPAddress)

# 构造 SAN 扩展（DNS + IP）
$SANParts = @("DNS:localhost")
$idx = 1
foreach ($ip in $IPs) {
    $SANParts += "IP:$ip"
    $idx++
}
$SANParts += "IP:127.0.0.1"
$SAN = $SANParts -join ","

Write-Host "生成自签名证书，SAN = $SAN"

# 检查 openssl 是否可用
$openssl = Get-Command openssl -ErrorAction SilentlyContinue
if (-not $openssl) {
    # 尝试 Git 自带的 openssl
    $gitOpenssl = "C:\Program Files\Git\usr\bin\openssl.exe"
    if (Test-Path $gitOpenssl) {
        $openssl = $gitOpenssl
    } else {
        Write-Host "错误：未找到 openssl。请安装 OpenSSL 或 Git for Windows 后重试。" -ForegroundColor Red
        Write-Host "也可以使用 Python 方式启动（见 run.py 中的 adhoc 模式）。" -ForegroundColor Yellow
        exit 1
    }
} else {
    $openssl = $openssl.Source
}

& $openssl req -x509 -newkey rsa:2048 -nodes `
    -keyout $KeyFile -out $CrtFile `
    -days 365 -subj "/CN=SpineUnified" `
    -addext "subjectAltName=$SAN"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "证书已生成：" -ForegroundColor Green
    Write-Host "  私钥: $KeyFile"
    Write-Host "  证书: $CrtFile"
    Write-Host ""
    Write-Host "启动服务时设置环境变量即可开启 HTTPS：" -ForegroundColor Cyan
    Write-Host '  $env:SPINE_SSL_CERT = "certs\server.crt"'
    Write-Host '  $env:SPINE_SSL_KEY  = "certs\server.key"'
    Write-Host "  python run.py"
} else {
    Write-Host "证书生成失败，请检查 openssl 输出。" -ForegroundColor Red
}
