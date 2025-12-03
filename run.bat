@echo off
REM E-Commerce Review Analyzer - Quick Launch Script

echo ================================================================================
echo E-COMMERCE REVIEW ANALYZER - QUICK LAUNCHER
echo ================================================================================
echo.

echo What would you like to do?
echo.
echo [1] Install dependencies
echo [2] Run dashboard
echo [3] Run accuracy test
echo [4] Complete setup (install + run)
echo [0] Exit
echo.

set /p choice="Enter your choice (0-4): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto run
if "%choice%"=="3" goto test
if "%choice%"=="4" goto complete
if "%choice%"=="0" goto exit
goto invalid

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Installation complete!
pause
goto end

:run
echo.
echo Starting Streamlit dashboard...
echo Dashboard will open at http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run app.py
goto end

:test
echo.
echo Running accuracy assessment...
python test_accuracy.py
echo.
pause
goto end

:complete
echo.
echo ================================================================================
echo COMPLETE SETUP
echo ================================================================================
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting dashboard...
echo Dashboard will open at http://localhost:8501
echo.
streamlit run app.py
goto end

:invalid
echo.
echo Invalid choice!
pause
goto end

:exit
echo.
echo Goodbye!
goto end

:end
