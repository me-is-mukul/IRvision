# Make the IRVision model server live from this laptop (free, no account needed).
#
# Starts the FastAPI model server (uses the local GPU) and exposes it to the internet
# through a Cloudflare quick tunnel. Prints the public server URL and the website link
# that switches the site to live mode.
#
# Usage (from the repository root, in PowerShell):
#   .\scripts\go_live.ps1 -Site https://your-site.vercel.app
# Stop with Ctrl+C (the server is stopped too).
#
# One-time setup:  winget install --id Cloudflare.cloudflared

param(
    [string]$Site = "",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Project environment not found at $python. See README -> Installation." -ForegroundColor Red
    exit 1
}
# find cloudflared: on PATH, or in the standard install folders (a terminal opened before the
# install does not see the updated PATH yet)
$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) {
    $cloudflared = @(
        "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
        "$env:ProgramFiles\cloudflared\cloudflared.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Links\cloudflared.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $cloudflared) {
    Write-Host "cloudflared is not installed. Run once:  winget install --id Cloudflare.cloudflared" -ForegroundColor Red
    exit 1
}

Write-Host "Starting the model server on port $Port ..."
$server = Start-Process -FilePath $python `
    -ArgumentList "-m", "uvicorn", "irvision.api.server:app", "--host", "127.0.0.1", "--port", "$Port" `
    -WorkingDirectory $root -PassThru -WindowStyle Minimized

try {
    $ready = $false
    for ($i = 0; $i -lt 60 -and -not $ready; $i++) {
        Start-Sleep -Seconds 2
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5
            $ready = $health.status -eq "ok"
        } catch { }
        if ($server.HasExited) { throw "The model server stopped unexpectedly." }
    }
    if (-not $ready) { throw "The model server did not start within 2 minutes." }
    Write-Host ("Model server ready (device: {0}, colorizer: {1})" -f $health.device, $health.colorizer) -ForegroundColor Green
    Write-Host "Opening the public tunnel ..."

    # cloudflared runs as its own process; its log (stderr) goes to a file we watch for the URL
    $tunnelLog = Join-Path $env:TEMP "irvision_tunnel_$Port.log"
    Remove-Item $tunnelLog -ErrorAction SilentlyContinue
    $tunnel = Start-Process -FilePath $cloudflared -ArgumentList "tunnel", "--url", "http://127.0.0.1:$Port" `
        -RedirectStandardError $tunnelLog -PassThru -WindowStyle Hidden

    $url = $null
    for ($i = 0; $i -lt 60 -and -not $url; $i++) {
        Start-Sleep -Seconds 2
        if ($tunnel.HasExited) { throw "cloudflared stopped unexpectedly. See $tunnelLog" }
        if (Test-Path $tunnelLog) {
            $match = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($match) { $url = $match.Matches[0].Value }
        }
    }
    if (-not $url) { throw "No tunnel address within 2 minutes. See $tunnelLog" }

    # a new tunnel address takes ~20-60 s to appear in DNS; announce only once it answers
    Write-Host "Tunnel created: $url"
    Write-Host "Waiting until it is reachable (usually under a minute) ..."
    $reachable = $false
    for ($i = 0; $i -lt 36 -and -not $reachable; $i++) {
        Start-Sleep -Seconds 5
        try {
            $null = Resolve-DnsName ([uri]$url).Host -DnsOnly -ErrorAction Stop
            $remote = Invoke-RestMethod -Uri "$url/api/health" -TimeoutSec 15
            $reachable = $remote.status -eq "ok"
        } catch { }
        if ($tunnel.HasExited) { throw "cloudflared stopped unexpectedly. See $tunnelLog" }
    }
    if (-not $reachable) {
        Write-Host "Could not reach the tunnel from this machine yet; it may still work for others shortly." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host " IRVision model server is LIVE" -ForegroundColor Cyan
    Write-Host "   Server:  $url" -ForegroundColor Cyan
    Write-Host "   Check:   $url/api/health" -ForegroundColor Cyan
    if ($Site) {
        Write-Host "   Website: $($Site.TrimEnd('/'))/?api=$url" -ForegroundColor Cyan
    } else {
        Write-Host "   Website: <your-site>/?api=$url" -ForegroundColor Cyan
    }
    Write-Host " Keep this window open. Press Ctrl+C to stop." -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan

    while (-not $tunnel.HasExited -and -not $server.HasExited) { Start-Sleep -Seconds 2 }
    Write-Host "The tunnel or the model server stopped." -ForegroundColor Yellow
}
finally {
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -Force
        Write-Host "Tunnel closed."
    }
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
        Write-Host "Model server stopped."
    }
}
