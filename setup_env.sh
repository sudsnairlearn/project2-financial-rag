#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "requirements.txt not found in $ROOT_DIR"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

PYTHON_EXE="$VENV_DIR/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
  echo "Python executable not found in $VENV_DIR/bin/python"
  exit 1
fi

echo "Upgrading pip..."
"$PYTHON_EXE" -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
"$PYTHON_EXE" -m pip install -r "$REQUIREMENTS_FILE"

echo "\nSetup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
