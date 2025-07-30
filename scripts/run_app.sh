#!/usr/bin/env bash
# Launch the FastAPI application using uvicorn

set -e

# Activate the environment (so that uvicorn and deps are in PATH)
source .venv/bin/activate

# Default host/port can be parameterized; using 0.0.0.0:8000 for accessibility in container/VM
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting FastAPI app on $HOST:$PORT ..."
# Using uvicorn to run the app; adjust workers if needed for concurrency.
uvicorn src.main:app --host $HOST --port $PORT --workers 4
