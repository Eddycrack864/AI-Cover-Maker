@echo off
setlocal
title AI Cover Maker
cd /d "%~dp0"

rem Extra arguments are passed through, e.g.: run-AI-Cover-Maker.bat --share --listen-port 8080
if exist env\python.exe (
    env\python.exe app.py --open %*
) else (
    echo Searching for global python and dependencies...
    where python > nul 2>&1
    if errorlevel 1 (
        echo Error: Python not found. Please install AI Cover Maker globally or run 'AI-Cover-Maker-installer.bat' to set up the environment.
        pause
        exit /b 1
    )
    python app.py --open %*
)

if errorlevel 1 (
    echo.
    echo AI Cover Maker exited with an error. Check the output above for details.
    pause
)
