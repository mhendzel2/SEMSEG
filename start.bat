@echo off
echo ==========================================
echo Starting SEMSEG...
echo ==========================================

REM Check if virtual environment exists
if not exist "venv" (
    echo Error: Virtual environment not found.
    echo Please run install.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the GUI application
echo Launching GUI...
python launch_gui.py

REM Deactivate virtual environment on exit
if %errorlevel% neq 0 (
    echo.
    echo Program exited with error code %errorlevel%.
    pause
)

deactivate
