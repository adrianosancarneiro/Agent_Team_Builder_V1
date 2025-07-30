#!/usr/bin/env bash
# Development setup script with dev dependencies

set -e

# Ensure uv is installed and environment is set up
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Run setup_env.sh first."
    exit 1
fi

# Ensure we're in project root directory (where pyproject.toml is)
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."  # move to project root relative to script directory

# Use uv to sync dependencies defined in pyproject.toml (including dev dependencies)
echo "Installing dependencies with dev tools using uv..."
uv sync

echo "Dev dependencies installation complete."
echo "You can now run tests with: uv run pytest"
echo "Format code with: uv run black src/"
echo "Sort imports with: uv run isort src/"
echo "Lint code with: uv run flake8 src/"
