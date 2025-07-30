#!/usr/bin/env bash
# Sync dependencies after updating LangGraph and AutoGen dependencies

set -e

# Ensure uv is installed and environment is set up
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Run setup_env.sh first."
    exit 1
fi

# Ensure we're in project root directory (where pyproject.toml is)
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."  # move to project root relative to script directory

# Activate the virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Use uv to sync dependencies defined in pyproject.toml
echo "Syncing LangGraph and AutoGen dependencies with uv..."
uv sync --no-dev

echo "Dependencies synchronization complete."
echo ""
echo "✅ Your environment now includes:"
echo "  - LangGraph for orchestration"
echo "  - AutoGen for agent implementation"
echo ""
echo "Run the tests with: pytest -q"
