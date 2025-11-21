#!/bin/bash
# Simple launcher script to ensure virtual environment is used

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    echo "Installing dependencies..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install numpy
    echo "✓ Setup complete!"
fi

# Run the main script
"$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
