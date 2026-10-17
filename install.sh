#!/usr/bin/env bash
#
# Install the Hermes Project Factory.
#
# This script only puts software on disk. It asks nothing, starts nothing,
# downloads no model weights and writes no configuration: it builds an
# isolated virtualenv for the factory, installs it, puts `hpf` and `hermes` on
# your PATH, and builds the neighbouring environment for the vLLM model server.
#
# Hermes and vLLM are the only software it installs. Hermes has no
# dependencies -- its YAML reader, test runner, build backend, GitHub client
# and weights downloader are its own -- so installing it downloads nothing.
# Anything else that would be useful is reimplemented rather than installed;
# see README "Dependencies".
#
# Choosing and starting a model server is the other half, and it lives in
# ./start.sh, which asks where the model is, which port to serve it on and how
# much context to reserve, writes your config and brings the server up.
#
#     ./install.sh && ./start.sh
#
# Usage: ./install.sh [--system] [--bin DIR] [--venv DIR]
#                     [--no-docker-image] [--no-vllm]
#
#   --system           link into /usr/local/bin instead of ~/.local/bin (sudo)
#   --bin DIR          link the commands into DIR
#   --venv DIR         build the virtualenv in DIR (default: <repo>/.venv)
#   --no-docker-image  skip pre-pulling the sandbox image
#   --no-vllm          skip the vLLM environment (you serve models yourself)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
VLLM_VENV="$REPO/.venv-vllm"
BIN_DIR="$HOME/.local/bin"
PULL_IMAGE=1
VLLM=1

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

# -- progress ----------------------------------------------------------------

# Several steps here take minutes, and a silent terminal during a 4 GB
# download is indistinguishable from a hang. `spin` covers work of unknown
# length; the step whose own tooling already prints real byte-level progress —
# pip installing vLLM — is deliberately left to print it, because a spinner
# that replaced "1.2/4.0 GB" with a twirling stick would be less information,
# not more.
#
# It degrades to a single plain line when stdout is not a terminal, so piping
# the install to a file does not produce a megabyte of carriage returns.
progress_tty() { [ -t 1 ]; }

spin() {  # spin <label> <command...>
  local label="$1"; shift
  local log pid start i=0 frames='|/-\'
  log="$(mktemp)"
  "$@" >"$log" 2>&1 &
  pid=$!
  start=$SECONDS
  if progress_tty; then
    while kill -0 "$pid" 2>/dev/null; do
      i=$(( (i + 1) % 4 ))
      printf '\r    %s %s  %ds\033[K' "${frames:i:1}" "$label" "$((SECONDS - start))"
      sleep 0.2
    done
    printf '\r\033[K'
  else
    printf '    %s ...\n' "$label"
  fi
  # The output is shown only on failure: on success it is noise, and on
  # failure it is the only thing that explains what happened.
  if wait "$pid"; then
    note "$label: done in $((SECONDS - start))s"
    rm -f "$log"
    return 0
  fi
  warn "$label: failed after $((SECONDS - start))s"
  tail -n 20 "$log" >&2
  rm -f "$log"
  return 1
}

# The interpreter vLLM is built against. It lags the newest CPython by a
# release or two and this host may well be running that one, so an older
# system interpreter is preferred when there is one.
find_python() {  # find_python <version...>
  local candidate
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --system)          BIN_DIR="/usr/local/bin" ;;
    --bin)             BIN_DIR="${2:?--bin needs a directory}"; shift ;;
    --venv)            VENV="${2:?--venv needs a directory}"; shift ;;
    --no-docker-image) PULL_IMAGE=0 ;;
    --no-vllm)         VLLM=0 ;;
    -h|--help)         sed -n '3,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# Offline on purpose: Hermes declares no dependencies and is built by the
# backend in build_backend/, so there is nothing for pip to fetch -- not even
# setuptools -- and --no-index makes a regression to that a loud failure
# instead of a quiet download.
install_hermes() { (cd "$REPO" && "$VENV/bin/python" -m pip install --quiet --no-index -e .); }
spin "installing hermes" install_hermes \
  || die "could not install hermes (pip 21.3 or newer is needed for an editable install)"

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

# -- tooling -----------------------------------------------------------------

note "checking tooling"
command -v git >/dev/null || warn "git not found: Hermes commits every milestone, install it"

if docker info >/dev/null 2>&1; then
  if [ "$PULL_IMAGE" -eq 1 ]; then
    IMAGE="$("$VENV/bin/python" -c \
      'from hermes.config import DEFAULTS; print(DEFAULTS["sandbox"]["image"])')"
    spin "pulling the sandbox image $IMAGE" docker pull -q "$IMAGE" \
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

# -- vllm --------------------------------------------------------------------

# The environment only. Which model it serves, on which port, with how much
# context is start.sh's business, and none of it is needed to install torch.
#
# vLLM's dependency tree is torch plus the CUDA runtime, several gigabytes of
# it, and it does not belong in the factory's own virtualenv. Only vLLM is
# installed into it: the weights are downloaded by Hermes itself
# (hermes/weights.py), not by a Hugging Face client installed alongside.
#
# Left unquiet on purpose: this is the single longest step of the install, and
# pip reports bytes and packages as it goes. That is better progress
# information than anything this script could draw over the top.
if [ "$VLLM" -eq 1 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no NVIDIA GPU detected. The published vLLM wheels need CUDA, and a"
    warn "CPU-only vLLM has to be built from source, so it is skipped. Point"
    warn "models.default at a server you run yourself — any OpenAI-compatible"
    warn "endpoint on this machine is supported — or pass --no-vllm to stop"
    warn "being told this."
    VLLM=0
  elif [ -x "$VLLM_VENV/bin/vllm" ]; then
    note "reusing the vLLM environment at $VLLM_VENV"
  else
    # vLLM publishes wheels for a window of CPython releases that trails the
    # newest one, so the releases inside that window are tried first and a
    # newer or unversioned python3 is the last resort.
    PY="$(find_python python3.12 python3.11 python3.10 python3.13 python3)" \
      || die "vLLM needs CPython 3.10 or newer; none was found"
    note "installing vLLM into $VLLM_VENV with $("$PY" -V 2>&1) (a few GB; progress below)"
    "$PY" -m venv "$VLLM_VENV" || die "could not create $VLLM_VENV"
    "$VLLM_VENV/bin/python" -m pip install --progress-bar on vllm \
      || die "could not install vLLM into $VLLM_VENV. If this interpreter is too new
    for the published wheels, install python3.12 and remove $VLLM_VENV."
  fi
fi

# -- next --------------------------------------------------------------------

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on your PATH. Add this to your shell profile:"
     warn "    export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

cat <<EOF

==> installed. Nothing is running yet and no config has been written.

Set up the model server and write your config:

    ./start.sh

It asks which model to serve, on which port, with how much context, writes
those answers to your config and brings the server up. Later runs of it
either ask again, defaulting to your current answers, or skip the questions
entirely and start what the config already describes:

    ./start.sh --config

Then build something, from any directory:

    mkdir -p ~/projects/my-idea && cd ~/projects/my-idea
    hpf "my-idea"
EOF
if [ "$VLLM" -eq 0 ]; then
  cat <<EOF

No vLLM environment was installed, so ./start.sh has no server to bring up.
Point models.default at an OpenAI-compatible server on this machine:

    hermes config --set models.default.base_url=http://localhost:11434/v1 \\
                  --set models.default.model=NAME
    hermes doctor
EOF
fi
