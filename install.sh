#!/usr/bin/env bash
#
# litesearch installer
#
# Usage:
#   ./install.sh              Install everything (core + ollama + server + gemini)
#   ./install.sh core         Install core only (bring your own embedder)
#   ./install.sh ollama       Core + Ollama embedder
#   ./install.sh server       Core + Ollama + REST API server
#   ./install.sh all          Everything
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${1:-all}"

echo "┌─────────────────────────────────────────────┐"
echo "│  litesearch installer                        │"
echo "│  profile: $PROFILE"
echo "└─────────────────────────────────────────────┘"
echo

# ── Find a Python with sqlite extension support ──────────────────────
# sqlite-vec needs enable_load_extension(). Many pyenv Pythons are compiled
# without it. We check Homebrew paths first (most reliable on macOS),
# then fall back to PATH.
find_compatible_python() {
    local candidates=(
        # Homebrew (macOS) — always compiled with extensions
        "/opt/homebrew/opt/python@3.13/bin/python3.13"
        "/opt/homebrew/opt/python@3.12/bin/python3.12"
        "/opt/homebrew/opt/python@3.11/bin/python3.11"
        "/usr/local/opt/python@3.13/bin/python3.13"
        "/usr/local/opt/python@3.12/bin/python3.12"
        # System / PATH
        "python3.13"
        "python3.12"
        "python3.11"
        "python3"
    )
    for py in "${candidates[@]}"; do
        local resolved=""
        if [[ "$py" == /* ]]; then
            [ -x "$py" ] && resolved="$py"
        else
            resolved="$(command -v "$py" 2>/dev/null || true)"
        fi
        [ -z "$resolved" ] && continue
        # Test if this Python can actually load sqlite extensions AND create a
        # working venv (some uv-managed Pythons fail at venv creation).
        if "$resolved" -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)" 2>/dev/null; then
            echo "$resolved"
            return 0
        fi
    done
    return 1
}

if [ -d ".venv" ]; then
    echo "→ Using existing .venv"
else
    echo "→ Finding Python with SQLite extension support..."
    if PYTHON=$(find_compatible_python); then
        echo "  Found: $PYTHON ($($PYTHON --version 2>&1))"
    else
        echo "  ERROR: No Python found with sqlite3.enable_load_extension() support."
        echo ""
        echo "  litesearch needs a Python compiled with --enable-loadable-sqlite-extensions."
        echo "  Fix options:"
        echo "    macOS:  brew install python@3.13"
        echo "    pyenv:  PYTHON_CONFIGURE_OPTS=\"--enable-loadable-sqlite-extensions\" pyenv install 3.13"
        echo "    Linux:  apt install python3-dev (usually works out of the box)"
        echo ""
        exit 1
    fi
    echo "→ Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

PY=".venv/bin/python"
PIP="$PY -m pip"
echo "→ Using: $($PY --version) at $PY"
echo

# Upgrade pip
$PIP install --upgrade pip --quiet

case "$PROFILE" in
    core)
        echo "→ Installing core dependencies..."
        $PIP install -r requirements-core.txt --quiet
        ;;
    ollama)
        echo "→ Installing core + Ollama..."
        $PIP install -r requirements-core.txt --quiet
        $PIP install "ollama>=0.3" --quiet
        ;;
    server)
        echo "→ Installing core + Ollama + REST API..."
        $PIP install -r requirements-core.txt --quiet
        $PIP install "ollama>=0.3" "fastapi>=0.110" "uvicorn[standard]>=0.29" --quiet
        ;;
    all)
        echo "→ Installing all dependencies..."
        $PIP install -r requirements.txt --quiet
        ;;
    *)
        echo "Unknown profile: $PROFILE"
        echo "Usage: ./install.sh [core|ollama|server|all]"
        exit 1
        ;;
esac

# Install litesearch itself in editable mode
echo "→ Installing litesearch (editable)..."
$PIP install -e . --quiet

# Verify sqlite-vec loads correctly
echo "→ Verifying sqlite-vec..."
if $PY -c "
import sqlite3, sqlite_vec
c = sqlite3.connect(':memory:')
c.enable_load_extension(True)
sqlite_vec.load(c)
print('  sqlite-vec loaded OK')
" 2>&1; then
    :
else
    echo "  WARNING: sqlite-vec failed to load. Search will not work."
    exit 1
fi

# Quick smoke test
echo "→ Running smoke test..."
$PY -c "
from litesearch import LiteSearch, __version__
print(f'  litesearch v{__version__} imported OK')
"

echo
echo "┌─────────────────────────────────────────────┐"
echo "│  Done.                                       │"
echo "│                                              │"
echo "│  Activate:  source .venv/bin/activate        │"
echo "│  Test:      python -m litesearch --help      │"
echo "│  Serve:     litesearch serve --db my.db      │"
echo "└─────────────────────────────────────────────┘"
