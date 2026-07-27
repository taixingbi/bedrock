#!/bin/bash
set -euo pipefail
exec python -m uvicorn app:app --host 0.0.0.0 --port "${AWS_LWA_PORT:-8080}"
