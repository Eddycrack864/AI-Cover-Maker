#!/bin/bash

# Shows what AI Cover Maker will run on: Python, torch and CUDA, onnxruntime, audio-separator and FFmpeg.
echo "AI Cover Maker Status"
echo

cd "$(dirname "$0")/.."

if [ -x "env/bin/python" ]; then
    if [ -z "$SSL_CERT_FILE" ] && [ -f "env/ssl/cert.pem" ]; then
        export SSL_CERT_FILE="$PWD/env/ssl/cert.pem"
    fi
    env/bin/python -c "import sys, torch, onnxruntime; cuda = torch.cuda.is_available(); print('Python', sys.version.split()[0]); print('torch', torch.__version__, '- CUDA', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if cuda else 'no NVIDIA GPU found, AI Cover Maker will run on the CPU'); print('onnxruntime providers:', ', '.join(onnxruntime.get_available_providers()))"
    echo
    env/bin/python -c "import sys; sys.argv = ['audio-separator', '-e']; from audio_separator.utils.cli import main; main()"
else
    echo "Error: the Python environment was not found. Run 'AI-Cover-Maker-installer.sh' first."
fi

echo
if command -v ffmpeg >/dev/null 2>&1; then
    echo "FFmpeg found: $(command -v ffmpeg)"
else
    echo "FFmpeg was not found. Install it with your package manager, e.g. sudo apt install ffmpeg"
fi

if [ -t 0 ]; then
    read -r -p "Press Enter to exit..."
fi
