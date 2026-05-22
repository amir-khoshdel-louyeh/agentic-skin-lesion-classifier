Write-Host "🧠 Starting OpenClaw + Ollama..." -ForegroundColor Cyan

# 1. Ollama check
$ollama = Get-Process ollama -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "✅ Ollama already running"
} else {
    Write-Host "⚠️ Starting Ollama..."
    Start-Process "ollama" "serve"
    Start-Sleep 3
}

# 2. OpenClaw gateway
$port = 18789
$check = netstat -ano | findstr $port

if ($check) {
    Write-Host "✅ OpenClaw gateway already running"
} else {
    Write-Host "⚠️ Starting OpenClaw gateway..."
    Start-Process "openclaw" "gateway run"
    Start-Sleep 5
}

# 3. Dashboard
openclaw dashboard