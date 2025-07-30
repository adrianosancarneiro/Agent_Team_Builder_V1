#!/usr/bin/env bash
# This script sets up a Python virtual environment using uv and installs uv itself if needed.

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing uv..."
    # Using pipx to install uv for isolation (could also use curl installer from Astral)
    python3 -m pip install --user pipx || { echo "Failed to install pipx"; exit 1; }
    pipx install uv
fi

# 2. Initialize a uv-managed virtual environment
# By default, uv looks for pyproject.toml in the current directory to resolve dependencies.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment with uv..."
    uv venv .venv  # create venv in .venv folder
fi

echo "Activating virtual environment..."
# Activate the environment for subsequent steps (if needed for interactive use)
source .venv/bin/activate || { echo "Activation failed"; exit 1; }

echo "Environment setup complete. uv is ready to manage dependencies."
