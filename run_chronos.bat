@echo off
echo 🏛️  Launching Chronos: The AI Historian...
echo ⏳ Please wait, initializing environment...

:: Navigate to the script's directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found. Please make sure you are in the project folder.
    pause
    exit /b
)

:: Check and start Ollama if not running
curl -s http://127.0.0.1:11434/api/tags > nul 2>&1
if errorlevel 1 (
    echo 🦙 Starting Ollama service...
    start /min "Ollama Service" ollama serve
    ping 127.0.0.1 -n 4 > nul
)

:: Activate venv and run Backend
echo 🚀 Starting AI Backend (FastAPI)...
start /min "Chronos Backend" cmd /k "call venv\Scripts\activate.bat && python server.py"

:: Wait for server to initialize
echo ⏳ Waiting for backend to warm up...
ping 127.0.0.1 -n 6 > nul

:: Run Frontend
echo 🎨 Starting UI (Gradio)...
call venv\Scripts\activate.bat
python app.py

pause
