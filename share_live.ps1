# ==============================================================================
# EchoCheck - Instant Public Sharing Script (Cloudflare Tunnel)
# Generates a Live Public HTTPS URL so anyone on the internet can test the website!
# ==============================================================================

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       EchoCheck - Launching Public Sharing               " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Flask server is already running
$connection = Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
if (-not $connection) {
    Write-Host "[1/2] Starting EchoCheck Flask Server on port 5000..." -ForegroundColor Green
    Start-Process -FilePath "py" -ArgumentList "app.py" -WindowStyle Minimized
    Start-Sleep -Seconds 3
} else {
    Write-Host "[1/2] EchoCheck server is running on port 5000." -ForegroundColor Green
}

# 2. Display Local Wi-Fi Address for classmates / mobile on same Wi-Fi
$localIP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.IPAddress -notlike "169.254*" } | Select-Object -First 1).IPAddress
Write-Host ""
Write-Host ">>> Same Wi-Fi Access (Classmates / Mobile Phone):" -ForegroundColor Magenta
Write-Host "    http://$($localIP):5000" -ForegroundColor White
Write-Host ""

# 3. Launching Cloudflare Tunnel
Write-Host "[2/2] Connecting to Global Cloudflare Edge Network..." -ForegroundColor Yellow
Write-Host "      Press Ctrl+C anytime to stop sharing." -ForegroundColor DarkGray
Write-Host ""

if (Test-Path ".\cloudflared.exe") {
    .\cloudflared.exe tunnel --url http://127.0.0.1:5000
} else {
    ssh -o StrictHostKeyChecking=no -R 80:localhost:5000 nokey@localhost.run
}
