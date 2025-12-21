@echo off
REM OpenFedLLM Windows Setup Script
REM This script sets up the environment variables needed for OpenFedLLM

echo ========================================
echo OpenFedLLM Windows Setup
echo ========================================
echo.

REM Set PYTHONPATH to include FinGPT evaluation directory
set PYTHONPATH=%PYTHONPATH%;%~dp0evaluation\FinGPT

echo PYTHONPATH set to include: %~dp0evaluation\FinGPT
echo.
echo Environment setup complete!
echo.
echo To make this permanent, you can add the following to your system environment variables:
echo PYTHONPATH = %PYTHONPATH%
echo.
echo Note: This environment variable is only set for the current session.
echo Run this script again if you open a new command prompt.
echo.

pause
