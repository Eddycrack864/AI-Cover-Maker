@echo off
title AI Cover Maker Status

rem Shows what AI Cover Maker will run on: Python, torch and CUDA, onnxruntime, audio-separator and FFmpeg.
cd /d "%~dp0\.."

if not exist "env\python.exe" goto :no_env

env\python.exe -c "import sys, torch, onnxruntime; cuda = torch.cuda.is_available(); print('Python', sys.version.split()[0]); print('torch', torch.__version__, '- CUDA', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if cuda else 'no NVIDIA GPU found, AI Cover Maker will run on the CPU'); print('onnxruntime providers:', ', '.join(onnxruntime.get_available_providers()))"
echo.
env\python.exe -c "import sys; sys.argv = ['audio-separator', '-e']; from audio_separator.utils.cli import main; main()"
goto :ffmpeg

:no_env
echo Error: the Python environment was not found. Run 'AI-Cover-Maker-installer.bat' first.

:ffmpeg
echo.
where ffmpeg > nul 2>&1
if errorlevel 1 (
    echo FFmpeg was not found on the PATH. Install it with info\ffmpeg-installer.bat
) else (
    echo FFmpeg found:
    where ffmpeg
)

echo.
pause
