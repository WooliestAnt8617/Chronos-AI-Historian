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

:: Activate venv and run Backend
echo 🚀 Starting AI Backend (FastAPI)...
start /min "Chronos Backend" cmd /k "venv\Scripts\activate.bat && python server.py"

:: Wait for server to initialize
echo ⏳ Waiting for backend to warm up...
timeout /t 5 /nobreak > nul

:: Run Frontend
echo 🎨 Starting UI (Gradio)...
call venv\Scripts\activate.bat
python app.py

pause
