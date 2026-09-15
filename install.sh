#!/usr/bin/env bash
#
# Install the Hermes Project Factory.
#
# Creates an isolated virtualenv for the factory, installs it and its single
# dependency, and puts `hpf` and `hermes` on your PATH. The optional OpenHands
# agent layer is installed alongside it, in an environment of its own. It never
# installs a model: Hermes is local-model-only and refuses to guess which model
# you want, so this script only reports what is missing and how to get it.
#
# Usage: ./install.sh [--dev] [--system] [--bin DIR] [--venv DIR]
#                     [--no-docker-image] [--no-config] [--no-openhands]
#
#   --dev              also install the test dependencies
#   --system           link into /usr/local/bin instead of ~/.local/bin (sudo)
#   --bin DIR          link the commands into DIR
#   --venv DIR         build the virtualenv in DIR (default: <repo>/.venv)
#   --no-docker-image  skip pre-pulling the sandbox image
#   --no-config        do not write ~/.config/hermes/config.yaml
#   --no-openhands     skip the optional OpenHands agent layer
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
OH_VENV="$REPO/.venv-openhands"
BIN_DIR="$HOME/.local/bin"
DEV=0
PULL_IMAGE=1
WRITE_CONFIG=1
OPENHANDS=1

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dev)             DEV=1 ;;
    --system)          BIN_DIR="/usr/local/bin" ;;
    --bin)             BIN_DIR="${2:?--bin needs a directory}"; shift ;;
    --venv)            VENV="${2:?--venv needs a directory}"; shift ;;
    --no-docker-image) PULL_IMAGE=0 ;;
    --no-config)       WRITE_CONFIG=0 ;;
    --no-openhands)    OPENHANDS=0 ;;
    -h|--help)         sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# -- python ------------------------------------------------------------------

note "checking python"
command -v python3 >/dev/null || die "python3 not found; install Python 3.10 or newer"
python3 - <<'PY' || die "Hermes needs Python 3.10 or newer; found $(python3 -V 2>&1)"
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY

# A virtualenv, not a --user install: many distributions now mark the system
# interpreter externally managed (PEP 668), where pip refuses to write at all.
if [ -x "$VENV/bin/python" ]; then
  note "reusing virtualenv at $VENV"
else
  note "creating virtualenv at $VENV"
  python3 -m venv "$VENV" \
    || die "could not create a virtualenv. On Debian/Ubuntu: sudo apt install python3-venv"
fi

"$VENV/bin/python" -m pip install --quiet --upgrade pip \
  || warn "could not upgrade pip; continuing with the bundled version"

note "installing hermes${DEV:+ }$([ "$DEV" -eq 1 ] && echo 'with test dependencies')"
if [ "$DEV" -eq 1 ]; then
  (cd "$REPO" && "$VENV/bin/python" -m pip install --quiet -e ".[dev]")
else
  (cd "$REPO" && "$VENV/bin/python" -m pip install --quiet -e ".")
fi

# -- commands ----------------------------------------------------------------

note "linking hpf and hermes into $BIN_DIR"
mkdir -p "$BIN_DIR" 2>/dev/null || sudo mkdir -p "$BIN_DIR"
for command in hpf hermes; do
  [ -x "$VENV/bin/$command" ] || die "$command was not installed into $VENV/bin"
  if [ -w "$BIN_DIR" ]; then
    ln -sf "$VENV/bin/$command" "$BIN_DIR/$command"
  else
    sudo ln -sf "$VENV/bin/$command" "$BIN_DIR/$command"
  fi
done

# -- configuration -----------------------------------------------------------

# `hpf` runs in project directories that have no hermes.yaml of their own, so
# the config in $HOME is what makes the command work from anywhere.
if [ "$WRITE_CONFIG" -eq 1 ]; then
  CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml"
  if [ -f "$CONFIG" ]; then
    note "keeping the existing config at $CONFIG"
  else
    mkdir -p "$(dirname "$CONFIG")"
    "$VENV/bin/hermes" config --sample > "$CONFIG"
    note "wrote a starter config to $CONFIG"
  fi
fi

# -- tooling -----------------------------------------------------------------

note "checking tooling"
command -v git >/dev/null || warn "git not found: Hermes commits every milestone, install it"
command -v gh  >/dev/null || note "gh not found (needed only for GitHub publishing)"

if docker info >/dev/null 2>&1; then
  if [ "$PULL_IMAGE" -eq 1 ]; then
    IMAGE="$("$VENV/bin/python" -c \
      'from hermes.config import DEFAULTS; print(DEFAULTS["sandbox"]["image"])')"
    note "pulling the sandbox image $IMAGE"
    docker pull -q "$IMAGE" >/dev/null \
      || warn "could not pull $IMAGE; Hermes will pull it on first run"
  fi
elif command -v docker >/dev/null; then
  warn "docker is installed but not usable by this user: start the daemon, or"
  warn "add yourself to the docker group (newgrp docker) and re-run this script"
  warn "without it generated code runs on this host with far weaker isolation"
else
  warn "docker not found: generated code would run unsandboxed on this host."
  warn "install docker before building anything you have not read."
fi

# -- openhands ---------------------------------------------------------------

# The optional agent layer of specification section 10, installed beside Hermes
# rather than into its virtualenv. Two reasons, both hard: the published CLI
# pins CPython 3.12, which is older than the interpreter Hermes itself runs on
# here, and its dependency tree carries hosted-provider SDKs that have no
# business in the factory's own environment. Hermes only ever invokes it as a
# subprocess, so a neighbouring install is the whole of the integration.
#
# Nothing is installed for the Feynman research workflow: it is part of Hermes
# itself (hermes/research.py), not a package.
if [ "$OPENHANDS" -eq 1 ]; then
  note "installing the OpenHands agent layer"
  if command -v openhands >/dev/null 2>&1; then
    note "openhands is already on PATH ($(command -v openhands))"
  elif command -v uv >/dev/null 2>&1; then
    # uv fetches CPython 3.12 itself, so this succeeds even on a host whose
    # only interpreter is too new for the package.
    uv tool install -q --python 3.12 openhands \
      || warn "uv could not install openhands; Hermes will use its own implementer"
  elif command -v python3.12 >/dev/null 2>&1; then
    if python3.12 -m venv "$OH_VENV" \
       && "$OH_VENV/bin/python" -m pip install --quiet --upgrade pip \
       && "$OH_VENV/bin/python" -m pip install --quiet openhands; then
      if [ -w "$BIN_DIR" ]; then
        ln -sf "$OH_VENV/bin/openhands" "$BIN_DIR/openhands"
      else
        sudo ln -sf "$OH_VENV/bin/openhands" "$BIN_DIR/openhands"
      fi
      note "installed openhands into $OH_VENV"
    else
      warn "could not install openhands into $OH_VENV"
    fi
  else
    warn "openhands requires CPython 3.12 and neither uv nor python3.12 was found."
    warn "install uv (https://docs.astral.sh/uv/) and re-run, or pass --no-openhands."
    warn "Hermes falls back to its own implementer either way."
  fi
  if command -v openhands >/dev/null 2>&1 || [ -x "$OH_VENV/bin/openhands" ]; then
    note "to use it, set 'enabled: true' under 'openhands' in your hermes.yaml"
  fi
fi

# -- models ------------------------------------------------------------------

note "checking the local inference server"
"$VENV/bin/hermes" doctor \
  || warn "doctor found problems above. Start your local model server (for example
     'ollama serve', then 'ollama pull qwen3:30b-a3b') and edit the config to
     name a model you actually have. Hermes will not fall back to a cloud API."

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on your PATH. Add this to your shell profile:"
     warn "    export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

cat <<'EOF'

==> installed.

Start a project from any directory:

    mkdir -p ~/projects/my-idea && cd ~/projects/my-idea
    hpf "my-idea"

Continue an interrupted project from the same directory:

    hpf

Watch it work:

    hermes status      # phase, board, progress
    hermes journal     # what each agent did
    hermes logs -f     # live log
EOF
