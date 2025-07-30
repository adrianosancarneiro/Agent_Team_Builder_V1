#!/usr/bin/env bash
# Install project dependencies using uv (assumes pyproject.toml defines them)

set -e

# Ensure uv is installed and environment is set up
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Run setup_env.sh first."
    exit 1
fi

# Ensure we're in project root directory (where pyproject.toml is)
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."  # move to project root relative to script directory

# Use uv to sync dependencies defined in pyproject.toml (production deps only, no dev)
echo "Installing dependencies with uv..."
uv sync --no-dev

echo "Dependencies installation complete."
