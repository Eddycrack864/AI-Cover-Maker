#!/bin/bash

echo -e "\e]2;AI Cover Maker\a"
cd "$(dirname "$0")"

# Extra arguments are passed through, e.g.: ./run-AI-Cover-Maker.sh --share --listen-port 8080
if [ -x "env/bin/python" ]; then
    PYTHON="env/bin/python"
    # OpenSSL looks for certificates where env/ was created, which breaks HTTPS if the folder was moved
    # (e.g. the precompiled version), so point it to the ones inside env/
    if [ -z "$SSL_CERT_FILE" ] && [ -f "env/ssl/cert.pem" ]; then
        export SSL_CERT_FILE="$PWD/env/ssl/cert.pem"
    fi
else
    echo "Searching for global python and dependencies..."
    if command -v python3 &> /dev/null; then
        PYTHON="python3"
    elif command -v python &> /dev/null; then
        PYTHON="python"
    else
        echo "Error: Python not found. Please install AI Cover Maker globally or run 'AI-Cover-Maker-installer.sh' to set up the environment."
        [ -t 0 ] && read -r -p "Press Enter to exit..."
        exit 1
    fi
fi

"$PYTHON" app.py --open "$@"
status=$?

if [ $status -ne 0 ]; then
    echo
    echo "AI Cover Maker exited with an error. Check the output above for details."
    [ -t 0 ] && read -r -p "Press Enter to exit..."
fi
exit $status
