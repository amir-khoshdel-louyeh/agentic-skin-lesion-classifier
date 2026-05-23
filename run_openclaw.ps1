# ===========================================================================
# USER CONFIGURATION
# ===========================================================================
# Enter your OpenClaw root folder path here:
$OPENCLAW_DIR   = "C:\Users\amirk\.openclaw"

# OpenClaw gateway executable filename:
$GATEWAY_FILE   = "gateway.cmd"

# Network port for OpenClaw gateway:
$OPENCLAW_PORT  = 18789

# Local language model process name (Ollama):
$OLLAMA_COMMAND = "ollama"
# ===========================================================================

# =========================
# OpenClaw + Ollama Launcher (SAFE VERSION)
# =========================

Clear-Host
Write-Host "Starting OpenClaw + Ollama..." -ForegroundColor Cyan

# Build the full gateway script path automatically
$gatewayScript = Join-Path $OPENCLAW_DIR $GATEWAY_FILE

# -------------------------
# Start Ollama
# -------------------------
$ollama = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue

if ($ollama) {
    Write-Host "Ollama already running." -ForegroundColor Green
} else {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    # Run Ollama in the background without opening a distracting window
    Start-Process $OLLAMA_COMMAND -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# -------------------------
# Start OpenClaw Gateway
# -------------------------
# Use native PowerShell command to check port status
$portActive = Get-NetTCPConnection -LocalPort $OPENCLAW_PORT -ErrorAction SilentlyContinue

if ($portActive) {
    Write-Host "OpenClaw gateway already running on port $OPENCLAW_PORT." -ForegroundColor Green
} else {
    if (Test-Path $gatewayScript) {
        Write-Host "Starting OpenClaw gateway from $OPENCLAW_DIR ..." -ForegroundColor Yellow
        
        # اجرای گیت‌وی دقیقاً در دایرکتوری کاری خودش تا کانفیگ‌ها را درست بخواند
        Start-Process "cmd.exe" -ArgumentList "/c `"$gatewayScript`"" -WorkingDirectory $OPENCLAW_DIR -WindowStyle Hidden
        Start-Sleep -Seconds 5
    } else {
        Write-Host "Warning: Gateway script not found at $gatewayScript" -ForegroundColor Red
        Write-Host "Please check your USER CONFIGURATION path at the top of this file." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "System is running. Type 'exit' to shutdown everything, or 'status' to check services." -ForegroundColor Green
Write-Host "---------------------------------------------------------------------------"

# =========================
# CLI LOOP
# =========================
while ($true) {

    $userInput = Read-Host "OpenClaw Shell"
    $trimmedInput = $userInput.Trim().ToLower()

    # -------------------------
    # EXIT (SAFE SHUTDOWN)
    # -------------------------
    if ($trimmedInput -eq "exit") {

        Write-Host "Shutting down system (safe mode)..." -ForegroundColor Red

        # =========================
        # 1. Stop OpenClaw Gracefully / Force Fallback
        # =========================
        Write-Host "Stopping OpenClaw runtime..." -ForegroundColor Yellow
        # Stop processes related to Node.js or processes with matching descriptions
        $openclawStill = Get-Process | Where-Object { $_.ProcessName -eq "node" -or $_.Description -like "*OpenClaw*" }
        if ($openclawStill) {
            $openclawStill | Stop-Process -Force -ErrorAction SilentlyContinue
        }

        # =========================
        # 2. Stop Ollama Gracefully & Fallback
        # =========================
        Write-Host "Stopping Ollama..." -ForegroundColor Yellow
        $ollamaProc = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue
        if ($ollamaProc) {
            $ollamaProc.CloseMainWindow() | Out-Null
            Start-Sleep -Seconds 2
            
            # If still open, force stop it
            $ollamaStill = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue
            if ($ollamaStill) {
                Stop-Process -Id $ollamaStill.Id -Force -ErrorAction SilentlyContinue
            }
        }

        Write-Host "All services stopped safely. Goodbye." -ForegroundColor Cyan
        break
    }

    # -------------------------
    # STATUS
    # -------------------------
    elseif ($trimmedInput -eq "status") {

        Write-Host "Checking system status..." -ForegroundColor Cyan

        # Check Ollama
        $ollamaProc = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue
        if ($ollamaProc) {
            Write-Host "   [+] Ollama: RUNNING (PID: $($ollamaProc.Id))" -ForegroundColor Green
        } else {
            Write-Host "   [-] Ollama: NOT RUNNING" -ForegroundColor Red
        }

        # Check gateway via network port
        $checkPort = Get-NetTCPConnection -LocalPort $OPENCLAW_PORT -ErrorAction SilentlyContinue
        if ($checkPort) {
            Write-Host "   [+] OpenClaw Gateway: RUNNING (Listening on port $OPENCLAW_PORT)" -ForegroundColor Green
        } else {
            Write-Host "   [-] OpenClaw Gateway: NOT RUNNING" -ForegroundColor Red
        }
    }

    # -------------------------
    # EMPTY INPUT
    # -------------------------
    elseif ([string]::IsNullOrWhiteSpace($trimmedInput)) {
        continue
    }

    # -------------------------
    # UNKNOWN COMMAND
    # -------------------------
    else {
        Write-Host "Unknown command: $userInput" -ForegroundColor Yellow
    }
}