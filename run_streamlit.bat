@echo off
REM TermExtractor Streamlit Web App Launcher for Windows
REM This script launches the Streamlit web interface

echo Starting TermExtractor Web Interface...
echo.

REM Check if ANTHROPIC_API_KEY is set
if "%ANTHROPIC_API_KEY%"=="" (
    echo Warning: ANTHROPIC_API_KEY environment variable not set.
    echo You can either:
    echo   1. Set it now: set ANTHROPIC_API_KEY=your-key-here
    echo   2. Enter it in the web interface when it opens
    echo.
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Streamlit not found. Installing...
    pip install streamlit plotly
    echo.
)

REM Launch Streamlit
echo Opening TermExtractor in your browser...
echo.
echo Note: You can stop the server anytime with Ctrl+C
echo.

streamlit run src/ui/streamlit_app.py ^
    --server.port 8501 ^
    --server.address localhost ^
    --server.headless false ^
    --browser.gatherUsageStats false
