#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENVS_DIR="$REPO_ROOT/envs"

if ! command -v uv &>/dev/null; then
    echo "uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"
echo ""

if [ -z "${1:-}" ]; then
    echo "Usage: ./setup.sh <group> [group ...]"
    echo ""
    echo "Available groups:"
    echo "  ai2thor       AI2-THOR and Multi-AI2-THOR environment"
    echo "  procthor      ProcTHOR and Multi-ProcTHOR environment"
    echo "  virtualhome   VirtualHome environment"
    echo "  carla         CARLA environment"
    echo "  embodiedcity  EmbodiedCity environment"
    echo "  game          Game environment"
    echo "  dev           Development/testing"
    echo ""
    echo "Example:"
    echo "  ./setup.sh ai2thor"
    echo "  ./setup.sh ai2thor dev"
    exit 1
fi

GROUPS=("$@")

echo "Setting up: ${GROUPS[*]}"
cd "$ENVS_DIR"
UV_ARGS=()
for group in "${GROUPS[@]}"; do
    UV_ARGS+=(--group "$group")
done
uv sync "${UV_ARGS[@]}"
echo ""
echo "Done. Activate with:"
echo "  source envs/.venv/bin/activate"
