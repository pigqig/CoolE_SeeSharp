#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.dotnet:${PATH}"
export DOTNET_ROOT="${HOME}/.dotnet"
cd "$(dirname "$0")/src/VisionStudio"
exec dotnet run --urls "${URLS:-http://0.0.0.0:43173}"
