@echo off
:: =================================================================
:: Fasal AI - Automated Launcher and Setup Script (v2)
:: =================================================================
:: This script will automatically set up the virtual environment,
:: install dependencies, compile translations, and run the app.
:: It is designed to be path-independent.
:: =================================================================
echo.
echo  Starting Fasal AI Launcher...
echo  ------------------------------
echo.

:: Check if python is installed and in PATH for the initial setup
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your system's PATH.
    echo Please install Python 3 and make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b
)

:: Define the name of the virtual environment directory
set VENV_DIR=fasal_ai_env_311

:: Check if the virtual environment directory exists.
if not exist "%VENV_DIR%" (
    echo [INFO] Virtual environment not found. Creating one now...
    python -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create the virtual environment. Please check your Python installation.
        pause
        exit /b
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment found.
)

echo.
echo [INFO] Installing/updating required packages from requirements.txt...
:: Use the python executable from the venv to run pip, which is the most reliable method
.\%VENV_DIR%\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages. Please check your requirements.txt file and internet connection.
    pause
    exit /b
)
echo [SUCCESS] All packages are installed.

echo.
echo [INFO] Compiling translation files...
:: Use the pybabel executable directly from the venv's Scripts folder
.\%VENV_DIR%\Scripts\pybabel compile -d translations
if %errorlevel% neq 0 (
    echo [WARNING] Could not compile translations. This is okay if no translation files exist yet.
) else (
    echo [SUCCESS] Translations compiled.
)

echo.
echo =============================================================
echo [INFO] Starting the Fasal AI application...
echo You can access the app at http://127.0.0.1:5000
echo Press CTRL+C in this window to stop the server.
echo =============================================================
echo.

:: Run the Flask application using the venv's python
.\%VENV_DIR%\Scripts\python.exe app.py

echo.
echo Server stopped.
pause

