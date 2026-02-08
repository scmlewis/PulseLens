@echo off
REM PulseLens Startup Script for Windows
REM Starts Flask backend and Streamlit frontend

echo ============================================
echo PulseLens - Customer Feedback Analyzer
echo ============================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJ_DIR=%SCRIPT_DIR:~0,-1%

REM Activate virtual environment if it exists
if exist "%PROJ_DIR%\.venv\Scripts\activate.bat" (
    call "%PROJ_DIR%\.venv\Scripts\activate.bat"
    echo ✓ Virtual environment activated
) else (
    echo ⚠ No virtual environment found. Using default Python.
)

echo.
echo Starting Flask backend on http://127.0.0.1:5000...
start "PulseLens Backend" cmd /k python "%PROJ_DIR%\backend\run.py"

echo Waiting for Flask to start (5 seconds)...
timeout /t 5 /nobreak

echo.
echo.
echo Starting Streamlit frontend on http://localhost:8501...
echo Press Ctrl+C to stop both services.
echo.

cd /d "%PROJ_DIR%"
streamlit run app.py

echo.
echo Stopping Flask backend...
taskkill /FI "WINDOWTITLE eq PulseLens Backend" /T /F >nul 2>&1

pause
