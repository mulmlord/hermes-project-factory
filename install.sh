#!/usr/bin/env bash
#
# Install the Hermes Project Factory.
#
# Creates an isolated virtualenv for the factory, installs it and its single
# dependency, and puts `hpf` and `hermes` on your PATH. The optional OpenHands
# agent layer is installed alongside it, in an environment of its own.
#
# It then sets up the model server Hermes talks to: vLLM, serving one local
# model on port 6767, under a systemd user unit that starts at boot and
# restarts on failure. You are asked once where the model is — a directory you
# already have, or a Hugging Face repository to download and where to keep it.
# Nothing is guessed silently and no hosted API is ever contacted.
#
# Usage: ./install.sh [--dev] [--system] [--bin DIR] [--venv DIR]
#                     [--no-docker-image] [--no-config] [--no-openhands]
#                     [--model REPO_OR_NAME] [--model-path DIR] [--models-dir DIR]
#                     [--port N] [--max-model-len N] [--gpu-util F]
#                     [--no-server] [--no-boot] [--force-config]
#
#   --dev              also install the test dependencies
#   --system           link into /usr/local/bin instead of ~/.local/bin (sudo)
#   --bin DIR          link the commands into DIR
#   --venv DIR         build the virtualenv in DIR (default: <repo>/.venv)
#   --no-docker-image  skip pre-pulling the sandbox image
#   --no-config        do not write ~/.config/hermes/config.yaml
#   --no-openhands     skip the optional OpenHands agent layer
#   --model REPO       Hugging Face repository to download and serve
#   --model-path DIR   serve a model directory you already have
#   --models-dir DIR   where downloaded weights are stored
#   --port N           port vLLM listens on (default: 6767)
#   --max-model-len N  context length vLLM serves (default: 16384)
#   --gpu-util F       fraction of VRAM vLLM may use (default: 0.90)
#   --no-server        install nothing for vLLM and start no server
#   --no-boot          install the unit but do not start it at boot
#   --force-config     rewrite an existing config to point at this server
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
OH_VENV="$REPO/.venv-openhands"
VLLM_VENV="$REPO/.venv-vllm"
BIN_DIR="$HOME/.local/bin"
DEV=0
PULL_IMAGE=1
WRITE_CONFIG=1
OPENHANDS=1

# -- model server defaults ---------------------------------------------------

SERVER=1
BOOT=1
FORCE_CONFIG=0
PORT=6767
MODEL_REPO=""
MODEL_PATH=""
MODELS_DIR="${HERMES_MODELS_DIR:-$HOME/.local/share/hermes/models}"
MAX_MODEL_LEN=16384
GPU_UTIL=0.90
UNIT_NAME="hermes-vllm.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

# A 7B AWQ coder: ~5 GiB of weights, which is what actually fits beside a
# desktop session on an 8 GiB card. vLLM keeps the whole model resident in
# VRAM — it has no CPU offload — so the ollama habit of naming a 30B model
# here would simply fail to start.
DEFAULT_REPO="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

# -- progress ----------------------------------------------------------------

# Several steps here take minutes, and a silent terminal during a 4 GB
# download is indistinguishable from a hang. Two indicators cover the two
# cases: `spin` for work of unknown length, `bar` for a wait with a deadline.
#
# Steps whose own tooling already prints real byte-level progress — uv and
# `hf download` — are deliberately left to print it, because a spinner that
# replaced "1.2/4.0 GB" with a twirling stick would be less information, not
# more.
#
# Both degrade to a single plain line when stdout is not a terminal, so piping
# the install to a file does not produce a megabyte of carriage returns.
BAR_FULL='############################'
BAR_EMPTY='............................'
BAR_WIDTH=28

progress_tty() { [ -t 1 ]; }

bar() {  # bar <done> <total> <label>
  progress_tty || return 0
  local done=$1 total=$2 label=$3 filled pct
  [ "$total" -gt 0 ] || total=1
  filled=$(( done * BAR_WIDTH / total ))
  [ "$filled" -gt "$BAR_WIDTH" ] && filled=$BAR_WIDTH
  pct=$(( done * 100 / total ))
  [ "$pct" -gt 100 ] && pct=100
  printf '\r    [%s%s] %3d%%  %s\033[K' \
    "${BAR_FULL:0:filled}" "${BAR_EMPTY:0:$((BAR_WIDTH - filled))}" "$pct" "$label"
}

bar_clear() { progress_tty && printf '\r\033[K' || true; }

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

while [ $# -gt 0 ]; do
  case "$1" in
    --dev)             DEV=1 ;;
    --system)          BIN_DIR="/usr/local/bin" ;;
    --bin)             BIN_DIR="${2:?--bin needs a directory}"; shift ;;
    --venv)            VENV="${2:?--venv needs a directory}"; shift ;;
    --no-docker-image) PULL_IMAGE=0 ;;
    --no-config)       WRITE_CONFIG=0 ;;
    --no-openhands)    OPENHANDS=0 ;;
    --model)           MODEL_REPO="${2:?--model needs a repository or name}"; shift ;;
    --model-path)      MODEL_PATH="${2:?--model-path needs a directory}"; shift ;;
    --models-dir)      MODELS_DIR="${2:?--models-dir needs a directory}"; shift ;;
    --port)            PORT="${2:?--port needs a number}"; shift ;;
    --max-model-len)   MAX_MODEL_LEN="${2:?--max-model-len needs a number}"; shift ;;
    --gpu-util)        GPU_UTIL="${2:?--gpu-util needs a fraction}"; shift ;;
    --no-server)       SERVER=0 ;;
    --no-boot)         BOOT=0 ;;
    --force-config)    FORCE_CONFIG=1 ;;
    -h|--help)         sed -n '3,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# -- configuration -----------------------------------------------------------

# `hpf` runs in project directories that have no hermes.yaml of their own, so
# the config in $HOME is what makes the command work from anywhere.
#
# Deferred until the model server section has run, because the config has to
# name the model that is actually being served and the port it answers on.
# MODEL_NAME is empty when no server was set up, and the sample's own defaults
# then stand unaltered.
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml"
MODEL_NAME=""

write_config() {
  [ "$WRITE_CONFIG" -eq 1 ] || return 0
  if [ -f "$CONFIG" ] && [ "$FORCE_CONFIG" -eq 0 ]; then
    note "keeping the existing config at $CONFIG"
    if [ -n "$MODEL_NAME" ]; then
      note "to point it at this server, set under models.default:"
      note "    backend: vllm"
      note "    base_url: http://localhost:$PORT/v1"
      note "    model: $MODEL_NAME"
      note "or re-run with --force-config to have it rewritten"
    fi
    return 0
  fi
  mkdir -p "$(dirname "$CONFIG")"
  # Patch the sample rather than restating it here, so the annotated file
  # stays the single description of every option.
  "$VENV/bin/hermes" config --sample \
    | sed -e "s|^\(    base_url: \).*|\1http://localhost:$PORT/v1|" \
          ${MODEL_NAME:+-e "s|^\(    model: \).*|\1$MODEL_NAME|"} \
          -e "s|^\(    context: \).*|\1$MAX_MODEL_LEN|" \
    > "$CONFIG"
  note "wrote a config to $CONFIG"
}

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

# -- model server ------------------------------------------------------------

# vLLM, one model, port 6767, under a systemd user unit. A unit rather than a
# shell command because the factory runs for hours and every agent call needs
# the server already up: an operator who has to remember to start it by hand is
# a dependency the autonomous loop cannot satisfy at 04:00.
#
# A *user* unit, not a system one, so nothing here needs root and the weights
# stay readable by you alone. The catch is that user units die with the last
# session unless lingering is on, which is what BOOT turns on below.
serve_model() {
  if [ "$SERVER" -eq 0 ]; then
    note "skipping the model server (--no-server)"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no NVIDIA GPU detected. The published vLLM wheels need CUDA, and a"
    warn "CPU-only vLLM has to be built from source, so the server is skipped."
    warn "Point models.default at a server you run yourself, or re-run with"
    warn "--no-server to stop being told this."
    return 0
  fi

  # Where the weights come from. Asked once, never guessed in the dark: an
  # unattended run that silently pulls several gigabytes is a worse outcome
  # than one that says what flag to pass.
  if [ -n "$MODEL_PATH" ]; then
    MODEL_PATH="$(cd "$MODEL_PATH" 2>/dev/null && pwd)" \
      || die "--model-path is not a directory that exists"
  elif [ -z "$MODEL_REPO" ]; then
    if [ -t 0 ]; then
      printf '\n'
      note "Which model should vLLM serve on port $PORT?"
      note "  1) download one from Hugging Face"
      note "  2) serve a model directory already on this machine"
      printf '   choice [1]: '
      read -r choice || choice=""
      if [ "${choice:-1}" = "2" ]; then
        printf '   path to the model directory: '
        read -r reply || reply=""
        [ -n "$reply" ] || die "no path given"
        MODEL_PATH="$(cd "${reply/#\~/$HOME}" 2>/dev/null && pwd)" \
          || die "not a directory: $reply"
      else
        printf '   Hugging Face repository [%s]: ' "$DEFAULT_REPO"
        read -r reply || reply=""
        MODEL_REPO="${reply:-$DEFAULT_REPO}"
        printf '   store the weights in [%s]: ' "$MODELS_DIR"
        read -r reply || reply=""
        MODELS_DIR="${reply:-$MODELS_DIR}"
        MODELS_DIR="${MODELS_DIR/#\~/$HOME}"
      fi
    else
      warn "no model chosen and no terminal to ask at. Re-run with --model"
      warn "REPO (downloaded into --models-dir) or --model-path DIR, or pass"
      warn "--no-server. Nothing was installed for vLLM."
      return 0
    fi
  fi

  # 8 GiB cards are the common case on laptops and the common cause of a
  # server that installs cleanly and then dies on first load.
  local vram
  vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "$vram" ] && [ "$vram" -lt 10000 ] 2>/dev/null; then
    note "GPU has ${vram} MiB of VRAM: serving at most a 7B 4-bit model"
    note "vLLM holds the whole model in VRAM, so a 30B model will not load"
  fi

  # vLLM's dependency tree is torch plus the CUDA runtime, several gigabytes of
  # it, and it does not belong in the factory's own virtualenv. It also has no
  # wheels for this host's CPython 3.14 yet, so the interpreter is pinned to
  # 3.12 the same way the OpenHands layer is.
  # Left unquiet on purpose: this is the single longest step of the install,
  # and both uv and pip report bytes and packages as they go. That is better
  # progress information than anything this script could draw over the top.
  note "installing vLLM into $VLLM_VENV (a few GB; progress below)"
  if [ -x "$VLLM_VENV/bin/vllm" ]; then
    note "reusing the vLLM environment at $VLLM_VENV"
  elif command -v uv >/dev/null 2>&1; then
    uv venv --quiet --python 3.12 "$VLLM_VENV" \
      || die "uv could not create a CPython 3.12 environment for vLLM"
    uv pip install --python "$VLLM_VENV/bin/python" vllm "huggingface_hub[cli]" \
      || die "could not install vLLM into $VLLM_VENV"
  else
    local py=""
    for candidate in python3.12 python3.11 python3.10; do
      command -v "$candidate" >/dev/null 2>&1 && { py="$candidate"; break; }
    done
    [ -n "$py" ] || die "vLLM needs CPython 3.10-3.12; install uv (https://docs.astral.sh/uv/) and re-run"
    "$py" -m venv "$VLLM_VENV" || die "could not create $VLLM_VENV"
    spin "upgrading pip in $VLLM_VENV" \
      "$VLLM_VENV/bin/python" -m pip install --quiet --upgrade pip || true
    "$VLLM_VENV/bin/python" -m pip install --progress-bar on vllm "huggingface_hub[cli]" \
      || die "could not install vLLM into $VLLM_VENV"
  fi

  if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$MODELS_DIR/$(basename "$MODEL_REPO")"
    # hf draws a per-file progress bar with real byte counts, so it is left to
    # speak for itself rather than being wrapped in a spinner.
    note "downloading $MODEL_REPO into $MODEL_PATH (progress below)"
    mkdir -p "$MODEL_PATH" || die "cannot write to $MODEL_PATH"
    "$VLLM_VENV/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_PATH" \
      || die "could not download $MODEL_REPO. Check the name, or pass --model-path."
  fi
  [ -f "$MODEL_PATH/config.json" ] \
    || warn "$MODEL_PATH has no config.json; vLLM may not recognise it"

  # The name the config will use, and the name vLLM will answer to, are made
  # the same string on purpose: preflight compares what is configured against
  # what /v1/models reports, and a path here would make that comparison fuzzy.
  MODEL_NAME="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"

  note "writing the $UNIT_NAME user unit"
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/$UNIT_NAME" <<EOF
[Unit]
Description=Hermes Project Factory model server (vLLM, port $PORT)
Documentation=https://docs.vllm.ai
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
# The weights are already on disk; serving must never reach the network.
Environment=HF_HUB_OFFLINE=1
ExecStart=$VLLM_VENV/bin/vllm serve $MODEL_PATH \\
    --served-model-name $MODEL_NAME \\
    --host 127.0.0.1 --port $PORT \\
    --max-model-len $MAX_MODEL_LEN \\
    --gpu-memory-utilization $GPU_UTIL
# vLLM drains in-flight requests on SIGINT; SIGTERM kills mid-generation.
KillSignal=SIGINT
Restart=always
RestartSec=10
TimeoutStartSec=900

[Install]
WantedBy=default.target
EOF

  systemctl --user daemon-reload || warn "could not reload the user systemd manager"

  if [ "$BOOT" -eq 1 ]; then
    # Without lingering a user unit stops at logout and never comes back until
    # the next interactive login, which is exactly the "start it by hand every
    # time" problem the unit exists to remove.
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then
      note "lingering is already enabled for $USER"
    elif loginctl enable-linger "$USER" 2>/dev/null; then
      note "enabled lingering for $USER, so the server starts at boot"
    elif sudo loginctl enable-linger "$USER"; then
      note "enabled lingering for $USER (as root)"
    else
      warn "could not enable lingering: the server will start when you log in,"
      warn "not at boot. Run: sudo loginctl enable-linger $USER"
    fi
    note "starting $UNIT_NAME"
    systemctl --user enable --now "$UNIT_NAME" \
      || warn "could not start $UNIT_NAME; see: systemctl --user status $UNIT_NAME"
  else
    note "installed $UNIT_NAME but left it stopped (--no-boot)"
    note "start it with: systemctl --user enable --now $UNIT_NAME"
    return 0
  fi

  # Loading several gigabytes onto the GPU takes minutes, and every command
  # after this one assumes the server answers, so wait rather than race it.
  # The bar runs against the timeout rather than against real progress: vLLM
  # reports nothing over HTTP until it is ready to serve, so elapsed-against-
  # deadline is the only honest total available. It is labelled as such.
  local waited=0 timeout=600 ready=0
  note "waiting for the server to load the model (up to $((timeout / 60)) minutes)"
  while [ "$waited" -lt "$timeout" ]; do
    if "$VENV/bin/python" - "$PORT" <<'PY'
import sys, urllib.error, urllib.request
try:
    with urllib.request.urlopen(
            f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=5) as resp:
        raise SystemExit(0 if resp.status == 200 else 1)
except (urllib.error.URLError, OSError):
    raise SystemExit(1)
PY
    then
      ready=1
      break
    fi
    sleep 5
    waited=$((waited + 5))
    bar "$waited" "$timeout" "loading $MODEL_NAME (${waited}s of ${timeout}s)"
  done
  bar_clear
  if [ "$ready" -eq 1 ]; then
    note "vLLM is serving $MODEL_NAME on http://localhost:$PORT/v1"
  else
    warn "the server did not answer on port $PORT within 10 minutes."
    warn "check: systemctl --user status $UNIT_NAME; journalctl --user -u $UNIT_NAME"
    warn "a model too large for this GPU is the usual cause; --max-model-len"
    warn "and --gpu-util are the knobs, a smaller model is the other answer."
  fi
}

serve_model
write_config

# -- models ------------------------------------------------------------------

note "checking the local inference server"
"$VENV/bin/hermes" doctor \
  || warn "doctor found problems above. Check that $UNIT_NAME is running and
     that the config names the model it serves. Hermes will not fall back to
     a cloud API."

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "$BIN_DIR is not on your PATH. Add this to your shell profile:"
     warn "    export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

cat <<EOF

==> installed.

The model server runs as a systemd user unit and comes back on boot:

    systemctl --user status $UNIT_NAME
    journalctl --user -u $UNIT_NAME -f     # what it is doing
    systemctl --user restart $UNIT_NAME    # after editing the unit

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
