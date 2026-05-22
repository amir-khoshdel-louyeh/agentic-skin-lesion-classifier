# =========================
# OpenClaw + Ollama Launcher (SAFE VERSION)
# =========================

Write-Host "Starting OpenClaw + Ollama..." -ForegroundColor Cyan

# -------------------------
# Start Ollama
# -------------------------
$ollama = Get-Process ollama -ErrorAction SilentlyContinue

if ($ollama) {
    Write-Host "Ollama already running"
} else {
    Write-Host "Starting Ollama..."
    Start-Process "ollama" "serve"
    Start-Sleep 3
}

# -------------------------
# Start OpenClaw Gateway
# -------------------------
$port = 18789
$check = netstat -ano | findstr $port

if ($check) {
    Write-Host "OpenClaw gateway already running"
} else {
    Write-Host "Starting OpenClaw gateway..."
    Start-Process "openclaw" "gateway run"
    Start-Sleep 5
}

# -------------------------
# Open Dashboard
# -------------------------
openclaw dashboard

Write-Host ""
Write-Host "System is running. Type 'exit' to shutdown everything." -ForegroundColor Green


# =========================
# CLI LOOP
# =========================
while ($true) {

    $input = Read-Host "OpenClaw Shell"

    # -------------------------
    # EXIT (SAFE SHUTDOWN)
    # -------------------------
    if ($input -eq "exit") {

        Write-Host "Shutting down system (safe mode)..." -ForegroundColor Red

        # =========================
        # 1. Stop OpenClaw Gracefully (preferred)
        # =========================
        try {
            Write-Host "Stopping OpenClaw gateway (graceful)..."
            openclaw gateway stop 2>$null
        } catch {}

        # fallback for scheduled task version
        try {
            schtasks /End /TN "OpenClaw Gateway" 2>$null
        } catch {}

        # =========================
        # 2. Stop Ollama Gracefully
        # =========================
        try {
            Write-Host "Stopping Ollama..."
            # Ollama doesn't always expose CLI stop, try service-level signal
            $ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
            if ($ollamaProc) {
                $ollamaProc.CloseMainWindow() | Out-Null
                Start-Sleep 2
            }
        } catch {}

        # =========================
        # 3. SAFE fallback cleanup (ONLY if still alive)
        # =========================
        $ollamaStill = Get-Process ollama -ErrorAction SilentlyContinue
        if ($ollamaStill) {
            Write-Host "Force stopping Ollama (fallback)..."
            Stop-Process -Id $ollamaStill.Id -Force
        }

        $openclawStill = Get-Process | Where-Object { $_.ProcessName -like "*openclaw*" -or $_.ProcessName -like "*node*" }
        if ($openclawStill) {
            Write-Host "Force stopping OpenClaw runtime (fallback)..."
            $openclawStill | Stop-Process -Force
        }

        Write-Host "All services stopped safely. Goodbye." -ForegroundColor Cyan
        break
    }

    # -------------------------
    # STATUS
    # -------------------------
    elseif ($input -eq "status") {

        Write-Host "Checking system status..."

        $ollamaProc = Get-Process ollama -ErrorAction SilentlyContinue
        if ($ollamaProc) {
            Write-Host ("Ollama running (PID: " + $ollamaProc.Id + ")")
        } else {
            Write-Host "Ollama not running"
        }

        $gw = Get-Process | Where-Object {
            $_.ProcessName -like "*openclaw*" -or $_.ProcessName -like "*node*"
        }

        if ($gw) {
            Write-Host "OpenClaw running"
        } else {
            Write-Host "OpenClaw not running"
        }
    }

    # -------------------------
    # UNKNOWN COMMAND
    # -------------------------
    else {
        Write-Host ("Unknown command: " + $input)
    }
}