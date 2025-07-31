#!/usr/bin/env bash
# Run Alembic database migrations

set -e

# Ensure we're in project root directory
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."  # move to project root relative to script directory

# Get environment variables (including database connection)
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    set -a
    source .env
    set +a
fi

# Run migrations
echo "Running database migrations..."
uv run alembic upgrade head

echo "Database migrations completed successfully."
