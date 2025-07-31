#!/usr/bin/env bash
# Run tests using pytest

set -e

# Ensure we're in project root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."  # move to project root relative to script directory

# Get environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

# Run pytest
echo "Running tests..."
uv run pytest "$@"

echo "Tests completed."
