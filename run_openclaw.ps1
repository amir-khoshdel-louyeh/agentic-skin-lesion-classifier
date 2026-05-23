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

# Python Infrastructure Configurations (CNN API Server)
# Set the project root directory and Python server script here.
# Use absolute paths so collaborators can edit this quickly.
$PROJECT_DIR    = "C:\Amir\GitHub\agentic-skin-lesion-classifier"
$PYTHON_SCRIPT  = "skin_agent.py"
$VENV_DIR       = Join-Path $PROJECT_DIR ".venv"

# Resolve the current script folder in case PROJECT_DIR is not valid.
$SCRIPT_DIR     = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path $PROJECT_DIR)) {
    $PROJECT_DIR = $SCRIPT_DIR
}
# ===========================================================================

# =========================
# OpenClaw + Ollama Launcher (SAFE VERSION)
# =========================

Clear-Host
Write-Host "Starting Agentic AI Stack Infrastructure..." -ForegroundColor Cyan

# Build the full gateway script path automatically
$gatewayScript = Join-Path $OPENCLAW_DIR $GATEWAY_FILE
$venvActivate  = Join-Path $PROJECT_DIR ".venv\Scripts\activate.bat"

# -------------------------
# 1. Start Ollama Core
# -------------------------
$ollama = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue

if ($ollama) {
    Write-Host "Ollama already running." -ForegroundColor Green
} else {
    Write-Host "Starting Ollama Core Service..." -ForegroundColor Yellow
    # Run Ollama in the background without opening a distracting window
    Start-Process $OLLAMA_COMMAND -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# -------------------------
# 2. Start OpenClaw Gateway
# -------------------------
$portActive = Get-NetTCPConnection -LocalPort $OPENCLAW_PORT -ErrorAction SilentlyContinue

if ($portActive) {
    Write-Host "OpenClaw gateway already running on port $OPENCLAW_PORT." -ForegroundColor Green
} else {
    if (Test-Path $gatewayScript) {
        Write-Host "Starting OpenClaw gateway in a new CMD window..." -ForegroundColor Yellow
        Start-Process "cmd.exe" -ArgumentList "/k `"$gatewayScript`"" -WorkingDirectory $OPENCLAW_DIR -WindowStyle Normal
        Start-Sleep -Seconds 4
    } else {
        Write-Host "Warning: Gateway script not found at $gatewayScript" -ForegroundColor Red
    }
}

# -------------------------
# 3. Start Python CNN FastAPI Server
# -------------------------
$scriptPath = Join-Path $PROJECT_DIR $PYTHON_SCRIPT

if (Test-Path $scriptPath) {
    Write-Host "Starting Python CNN API Server ($PYTHON_SCRIPT) in a new CMD window..." -ForegroundColor Yellow
    
    # Build command arguments: switch to the project folder, activate the venv, and run the Python server.
    # Add a window title so users can easily identify the server terminal.
    $cmdArguments = "/k title Skin Agent Server && cd /d `"$PROJECT_DIR`" && call `"$venvActivate`" && python `"$PYTHON_SCRIPT`""
    
    Start-Process "cmd.exe" -ArgumentList $cmdArguments -WindowStyle Normal
    Start-Sleep -Seconds 3
} else {
    Write-Host "Warning: Python server script not found at $scriptPath" -ForegroundColor Red
    Write-Host "Please check your PROJECT_DIR variable at the top of this file." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "All components requested. Type 'exit' to shutdown everything, or 'status' to check services." -ForegroundColor Green
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

        Write-Host "Shutting down system infrastructure (safe mode)..." -ForegroundColor Red

        # =========================
        # 1. Stop OpenClaw Runtime
        # =========================
        Write-Host "Stopping OpenClaw runtime and closing gateway window..." -ForegroundColor Yellow
        $openclawStill = Get-Process | Where-Object { $_.ProcessName -eq "node" -or $_.Description -like "*OpenClaw*" -or $_.MainWindowTitle -like "*gateway.cmd*" }
        if ($openclawStill) {
            $openclawStill | Stop-Process -Force -ErrorAction SilentlyContinue
        }

        # =========================
        # 2. Stop Python FastAPI Server
        # =========================
        Write-Host "Stopping Python FastAPI Server and closing terminal..." -ForegroundColor Yellow
        # بستن پروسس‌های مربوط به پایتون که با این نام اسکریپت یا تایتل باز شده‌اند
        $pythonStill = Get-Process | Where-Object { $_.ProcessName -eq "python" -or $_.MainWindowTitle -like "*skin_agent.py*" -or $_.MainWindowTitle -like "*cmd.exe*" }
        if ($pythonStill) {
            $pythonStill | Stop-Process -Force -ErrorAction SilentlyContinue
        }

        # =========================
        # 3. Stop Ollama Core
        # =========================
        Write-Host "Stopping Ollama..." -ForegroundColor Yellow
        $ollamaProc = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue
        if ($ollamaProc) {
            $ollamaProc.CloseMainWindow() | Out-Null
            Start-Sleep -Seconds 2
            
            $ollamaStill = Get-Process $OLLAMA_COMMAND -ErrorAction SilentlyContinue
            if ($ollamaStill) {
                Stop-Process -Id $ollamaStill.Id -Force -ErrorAction SilentlyContinue
            }
        }

        Write-Host "All services stopped safely. Runtime cleaned. Goodbye." -ForegroundColor Cyan
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

        # Check Python API
        $pythonProc = Get-Process | Where-Object { $_.ProcessName -eq "python" -and $_.MainWindowTitle -like "*skin_agent.py*" }
        if ($pythonProc) {
            Write-Host "   [+] Python FastAPI: RUNNING (PID: $($pythonProc[0].Id))" -ForegroundColor Green
        } else {
            # بررسی ثانویه از روی پورت پیش‌فرض FastAPI (8000)
            $checkApiPort = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
            if ($checkApiPort) {
                Write-Host "   [+] Python FastAPI: RUNNING (Active on Port 8000)" -ForegroundColor Green
            } else {
                Write-Host "   [-] Python FastAPI: NOT RUNNING" -ForegroundColor Red
            }
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