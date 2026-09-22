#!/usr/bin/env bash
#
# Install the Hermes Project Factory.
#
# This script only puts software on disk. It asks nothing, starts nothing,
# downloads no model weights and writes no configuration: it builds an
# isolated virtualenv for the factory, installs it and its single dependency,
# puts `hpf` and `hermes` on your PATH, and builds the neighbouring
# environments for vLLM and the optional OpenHands agent layer.
#
# Choosing and starting a model server is the other half, and it lives in
# ./start.sh, which asks where the model is, which port to serve it on and how
# much context to reserve, writes your config and brings the server up.
#
#     ./install.sh && ./start.sh
#
# Usage: ./install.sh [--dev] [--system] [--bin DIR] [--venv DIR]
#                     [--no-docker-image] [--no-openhands] [--no-vllm]
#                     [--classifier]
#
#   --dev              also install the test dependencies
#   --system           link into /usr/local/bin instead of ~/.local/bin (sudo)
#   --bin DIR          link the commands into DIR
#   --venv DIR         build the virtualenv in DIR (default: <repo>/.venv)
#   --no-docker-image  skip pre-pulling the sandbox image
#   --no-openhands     skip the optional OpenHands agent layer
#   --no-vllm          skip the vLLM environment (you serve models yourself)
#   --classifier       also install the security-gate classifier (torch)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
OH_VENV="$REPO/.venv-openhands"
VLLM_VENV="$REPO/.venv-vllm"
LAYA_VENV="$REPO/.venv-laya"
BIN_DIR="$HOME/.local/bin"
DEV=0
PULL_IMAGE=1
OPENHANDS=1
VLLM=1

# Off unless asked for. It is a second opinion on the security gate, it costs
# a torch install, and an un-tuned checkpoint is a source of noise rather than
# of findings — so nobody gets it by accident.
CLASSIFIER=0

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

# -- progress ----------------------------------------------------------------

# Several steps here take minutes, and a silent terminal during a 4 GB
# download is indistinguishable from a hang. `spin` covers work of unknown
# length; steps whose own tooling already prints real byte-level progress —
# uv and pip — are deliberately left to print it, because a spinner that
# replaced "1.2/4.0 GB" with a twirling stick would be less information, not
# more.
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

# The interpreter vLLM, OpenHands and the classifier are built against. All
# three lag the newest CPython by a release or two, and this host may well be
# running it, so uv fetching 3.12 is the reliable answer and a system 3.10-3.12
# the fallback.
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
    --dev)             DEV=1 ;;
    --system)          BIN_DIR="/usr/local/bin" ;;
    --bin)             BIN_DIR="${2:?--bin needs a directory}"; shift ;;
    --venv)            VENV="${2:?--venv needs a directory}"; shift ;;
    --no-docker-image) PULL_IMAGE=0 ;;
    --no-openhands)    OPENHANDS=0 ;;
    --no-vllm)         VLLM=0 ;;
    --classifier)      CLASSIFIER=1 ;;
    -h|--help)         sed -n '3,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

spin "upgrading pip" "$VENV/bin/python" -m pip install --quiet --upgrade pip \
  || warn "could not upgrade pip; continuing with the bundled version"

# A function because spin runs its argument list as one command, and this one
# needs a subshell to keep the directory change local.
install_hermes() { (cd "$REPO" && "$VENV/bin/python" -m pip install --quiet -e "$1"); }
if [ "$DEV" -eq 1 ]; then
  spin "installing hermes with test dependencies" install_hermes ".[dev]" \
    || die "could not install hermes"
else
  spin "installing hermes" install_hermes "." || die "could not install hermes"
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

# -- tooling -----------------------------------------------------------------

note "checking tooling"
command -v git >/dev/null || warn "git not found: Hermes commits every milestone, install it"
command -v gh  >/dev/null || note "gh not found (needed only for GitHub publishing)"

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
    spin "installing openhands" uv tool install -q --python 3.12 openhands \
      || warn "uv could not install openhands; Hermes will use its own implementer"
  elif command -v python3.12 >/dev/null 2>&1; then
    build_openhands() {
      python3.12 -m venv "$OH_VENV" \
        && "$OH_VENV/bin/python" -m pip install --quiet --upgrade pip \
        && "$OH_VENV/bin/python" -m pip install --quiet openhands
    }
    if spin "installing openhands into $OH_VENV" build_openhands; then
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

# -- security classifier -----------------------------------------------------

# A second opinion on the security gate: the pattern list in hermes/verify.py
# reads what it was told to look for, and a classifier covers some of what no
# regex expresses. Installed beside Hermes for the same reason vLLM is — it
# wants torch and transformers, and the factory's own environment has exactly
# one dependency and should keep it.
#
# Only the environment is built here. start.sh writes its unit and starts it,
# because which checkpoint it serves and on which port are questions about a
# running server rather than about an install.
if [ "$CLASSIFIER" -eq 1 ]; then
  note "installing the security-gate classifier"
  build_classifier() {
    if command -v uv >/dev/null 2>&1; then
      uv venv --quiet --python 3.12 "$LAYA_VENV" \
        && uv pip install --quiet --python "$LAYA_VENV/bin/python" laya
    else
      local py
      py="$(find_python python3.12 python3.11 python3.10)" || return 1
      "$py" -m venv "$LAYA_VENV" \
        && "$LAYA_VENV/bin/python" -m pip install --quiet --upgrade pip \
        && "$LAYA_VENV/bin/python" -m pip install --quiet laya
    fi
  }
  if [ -x "$LAYA_VENV/bin/python" ] \
     && "$LAYA_VENV/bin/python" -c 'import laya' 2>/dev/null; then
    note "reusing the classifier environment at $LAYA_VENV"
  elif spin "installing laya into $LAYA_VENV (torch; a few GB)" build_classifier; then
    note "installed the classifier into $LAYA_VENV"
  else
    warn "could not install the classifier; the security gate keeps its patterns"
    warn "it needs CPython 3.10-3.12 — install uv (https://docs.astral.sh/uv/) and re-run"
    CLASSIFIER=0
  fi
fi

# -- vllm --------------------------------------------------------------------

# The environment only. Which model it serves, on which port, with how much
# context is start.sh's business, and none of it is needed to install torch.
#
# vLLM's dependency tree is torch plus the CUDA runtime, several gigabytes of
# it, and it does not belong in the factory's own virtualenv. It also has no
# wheels for the newest CPython releases, so the interpreter is pinned to 3.12
# the same way the OpenHands layer is.
#
# Left unquiet on purpose: this is the single longest step of the install, and
# both uv and pip report bytes and packages as they go. That is better progress
# information than anything this script could draw over the top.
if [ "$VLLM" -eq 1 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no NVIDIA GPU detected. The published vLLM wheels need CUDA, and a"
    warn "CPU-only vLLM has to be built from source, so it is skipped. Point"
    warn "models.default at a server you run yourself — Ollama, llama.cpp and"
    warn "any other OpenAI-compatible endpoint are fully supported — or pass"
    warn "--no-vllm to stop being told this."
    VLLM=0
  elif [ -x "$VLLM_VENV/bin/vllm" ]; then
    note "reusing the vLLM environment at $VLLM_VENV"
  else
    note "installing vLLM into $VLLM_VENV (a few GB; progress below)"
    if command -v uv >/dev/null 2>&1; then
      uv venv --quiet --python 3.12 "$VLLM_VENV" \
        || die "uv could not create a CPython 3.12 environment for vLLM"
      uv pip install --python "$VLLM_VENV/bin/python" vllm "huggingface_hub[cli]" \
        || die "could not install vLLM into $VLLM_VENV"
    else
      PY="$(find_python python3.12 python3.11 python3.10)" \
        || die "vLLM needs CPython 3.10-3.12; install uv (https://docs.astral.sh/uv/) and re-run"
      "$PY" -m venv "$VLLM_VENV" || die "could not create $VLLM_VENV"
      spin "upgrading pip in $VLLM_VENV" \
        "$VLLM_VENV/bin/python" -m pip install --quiet --upgrade pip || true
      "$VLLM_VENV/bin/python" -m pip install --progress-bar on vllm "huggingface_hub[cli]" \
        || die "could not install vLLM into $VLLM_VENV"
    fi
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
Point models.default at an OpenAI-compatible server you run yourself:

    hermes config --set models.default.base_url=http://localhost:11434/v1 \\
                  --set models.default.model=NAME
    hermes doctor
EOF
fi
