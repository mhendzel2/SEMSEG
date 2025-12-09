@echo off
echo ==========================================
echo SEMSEG Installation Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.8 or higher from python.org.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    echo Warning: requirements.txt not found. Skipping dependency installation.
)

REM Install the package in editable mode
if exist "setup.py" (
    echo Installing SEMSEG in editable mode...
    pip install -e .
    if %errorlevel% neq 0 (
        echo Error: Failed to install package.
        pause
        exit /b 1
    )
)

echo.
echo ==========================================
echo Installation complete!
echo You can now run the program using start.bat
echo ==========================================
pause
