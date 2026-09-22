#!/usr/bin/env bash
#
# Set up and start the model server the Hermes Project Factory talks to.
#
# ./install.sh puts the software on disk and stops there. This script is the
# half that makes a decision: it asks which model to serve, from where, on
# which port and with how much context, writes those answers into your config,
# writes the systemd user unit that serves them, and brings the server up.
#
# Run it again whenever those answers change — every question is pre-filled
# with what your config says now, so it doubles as an editor for them. With
# --config it asks nothing at all and starts exactly what the config already
# describes, which is the form to use from a script or a login item.
#
# Usage: ./start.sh [--config] [--config-file PATH]
#                   [--model REPO] [--model-path DIR] [--models-dir DIR]
#                   [--port N] [--context N] [--gpu-util F]
#                   [--classifier] [--no-classifier]
#                   [--classifier-model REPO] [--classifier-port N]
#                   [--boot] [--no-wait] [--stop]
#
#   --config           ask nothing: serve what the config already names
#   --config-file PATH the config to read and update
#   --model REPO       Hugging Face repository to download and serve
#   --model-path DIR   serve a model directory you already have
#   --models-dir DIR   where downloaded weights are stored
#   --port N           port vLLM listens on
#   --context N        context length served, and the config's `context`
#   --gpu-util F       fraction of VRAM vLLM may use
#   --classifier       also run the security-gate classifier
#   --no-classifier    do not run it, and turn it off in the config
#   --classifier-model REPO  checkpoint it serves
#   --classifier-port N      port it listens on
#   --boot             also run the server at boot, not just during a run
#   --no-wait          do not wait for the weights to finish loading
#   --stop             stop the server instead of starting it
#
# Anything given as a flag is not asked about. Given every answer, or run
# without a terminal, it asks nothing and uses your config's values for the
# rest.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
VLLM_VENV="$REPO/.venv-vllm"
LAYA_VENV="$REPO/.venv-laya"
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml"
UNIT_NAME="hermes-vllm.service"
CLASSIFIER_UNIT="hermes-classifier.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

FROM_CONFIG=0
BOOT=0
WAIT=1
STOP=0
CLASSIFIER=-1          # -1: whatever the config says; 0/1: --no-classifier/--classifier

# Answers. Empty means "not given on the command line", which is what makes a
# question worth asking.
MODEL_REPO=""
MODEL_PATH=""
MODELS_DIR=""
PORT=""
CONTEXT=""
GPU_UTIL=""
CLASSIFIER_REPO=""
CLASSIFIER_PORT=""

# A 7B AWQ coder: ~5 GiB of weights, which is what actually fits beside a
# desktop session on an 8 GiB card. vLLM keeps the whole model resident in
# VRAM — it has no CPU offload — so the ollama habit of naming a 30B model
# here would simply fail to start.
DEFAULT_REPO="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
DEFAULT_MODELS_DIR="${HERMES_MODELS_DIR:-$HOME/.local/share/hermes/models}"

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

BAR_FULL='############################'
BAR_EMPTY='............................'
BAR_WIDTH=28

bar() {  # bar <done> <total> <label>
  [ -t 1 ] || return 0
  local done=$1 total=$2 label=$3 filled pct
  [ "$total" -gt 0 ] || total=1
  filled=$(( done * BAR_WIDTH / total ))
  [ "$filled" -gt "$BAR_WIDTH" ] && filled=$BAR_WIDTH
  pct=$(( done * 100 / total ))
  [ "$pct" -gt 100 ] && pct=100
  printf '\r    [%s%s] %3d%%  %s\033[K' \
    "${BAR_FULL:0:filled}" "${BAR_EMPTY:0:$((BAR_WIDTH - filled))}" "$pct" "$label"
}

bar_clear() { [ -t 1 ] && printf '\r\033[K' || true; }

while [ $# -gt 0 ]; do
  case "$1" in
    --config)          FROM_CONFIG=1 ;;
    --config-file)     CONFIG="${2:?--config-file needs a path}"; shift ;;
    --model)           MODEL_REPO="${2:?--model needs a repository}"; shift ;;
    --model-path)      MODEL_PATH="${2:?--model-path needs a directory}"; shift ;;
    --models-dir)      MODELS_DIR="${2:?--models-dir needs a directory}"; shift ;;
    --port)            PORT="${2:?--port needs a number}"; shift ;;
    --context|--max-model-len)
                       CONTEXT="${2:?--context needs a number}"; shift ;;
    --gpu-util)        GPU_UTIL="${2:?--gpu-util needs a fraction}"; shift ;;
    --classifier)      CLASSIFIER=1 ;;
    --no-classifier)   CLASSIFIER=0 ;;
    --classifier-model) CLASSIFIER=1; CLASSIFIER_REPO="${2:?--classifier-model needs a repository}"; shift ;;
    --classifier-port) CLASSIFIER=1; CLASSIFIER_PORT="${2:?--classifier-port needs a number}"; shift ;;
    --boot)            BOOT=1 ;;
    --no-wait)         WAIT=0 ;;
    --stop)            STOP=1 ;;
    -h|--help)         sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

[ -x "$VENV/bin/hermes" ] || die "Hermes is not installed here. Run ./install.sh first."

# -- stopping ----------------------------------------------------------------

# The counterpart to the whole script, because the one thing you want after
# "start it and keep it resident" is a way to hand the VRAM back.
if [ "$STOP" -eq 1 ]; then
  for unit in "$UNIT_NAME" "$CLASSIFIER_UNIT"; do
    if [ -f "$UNIT_DIR/$unit" ] && systemctl --user is-active --quiet "$unit"; then
      note "stopping $unit"
      systemctl --user stop "$unit" || warn "could not stop $unit"
    fi
  done
  exit 0
fi

# -- what the config says now ------------------------------------------------

# Every question below is pre-filled from here, so re-running this script edits
# your setup rather than starting it over, and --config skips the asking
# entirely because these values are already the answers.
eval "$("$VENV/bin/python" - "$CONFIG" <<'PY'
import copy, shlex, sys
from pathlib import Path
from urllib.parse import urlparse

from hermes.config import DEFAULTS, Config

path = Path(sys.argv[1])
config = Config.load(path) if path.is_file() else Config(copy.deepcopy(DEFAULTS))
model = config.get("models", {}).get("default", {})
classifier = config.get("security.classifier", {})


def emit(name, value):
    print(f"{name}={shlex.quote(str(value))}")


emit("HAVE_CONFIG", int(path.is_file()))
emit("CFG_PORT", urlparse(model.get("base_url", "")).port or 6767)
emit("CFG_CONTEXT", model.get("context") or 16384)
emit("CFG_MODEL", model.get("model") or "")
emit("CFG_MODEL_PATH", config.get("server.model_path") or "")
emit("CFG_GPU_UTIL", config.get("server.gpu_util") or 0.90)
emit("CFG_UNIT", config.get("server.unit") or "hermes-vllm.service")
emit("CFG_WAIT", config.get("server.wait") or 600)
emit("CFG_CLASSIFIER", int(bool(classifier.get("enabled"))))
emit("CFG_CLASSIFIER_PORT", urlparse(classifier.get("base_url", "")).port or 6768)
emit("CFG_CLASSIFIER_MODEL", classifier.get("checkpoint") or "convaiinnovations/laya")
PY
)" || die "could not read $CONFIG"

UNIT_NAME="$CFG_UNIT"
[ "$HAVE_CONFIG" -eq 1 ] && note "reading $CONFIG" || note "no config yet; starting from the defaults"

# --config is a promise that nothing is asked and nothing is rewritten, so it
# needs the config to already name a model directory. It will not, exactly
# once: the first run.
if [ "$FROM_CONFIG" -eq 1 ]; then
  [ -n "$CFG_MODEL_PATH" ] || die \
    "$CONFIG does not say which weights to serve (server.model_path is empty).
    Run ./start.sh once without --config to answer that, or pass --model-path."
  MODEL_PATH="${MODEL_PATH:-$CFG_MODEL_PATH}"
  PORT="${PORT:-$CFG_PORT}"
  CONTEXT="${CONTEXT:-$CFG_CONTEXT}"
  GPU_UTIL="${GPU_UTIL:-$CFG_GPU_UTIL}"
  [ "$CLASSIFIER" -eq -1 ] && CLASSIFIER="$CFG_CLASSIFIER"
fi

# -- asking ------------------------------------------------------------------

# Only ever asked at a terminal. Without one — a login item, a cron entry, a
# piped install — the config's own values stand, because a script that blocks
# on a prompt nobody can see is a hang, not a question.
interactive() { [ "$FROM_CONFIG" -eq 0 ] && [ -t 0 ]; }

ask() {  # ask <prompt> <default>
  local prompt="$1" default="$2" reply
  if ! interactive; then
    printf '%s\n' "$default"
    return 0
  fi
  printf '   %s [%s]: ' "$prompt" "$default" >&2
  read -r reply || reply=""
  printf '%s\n' "${reply:-$default}"
}

ask_valid() {  # ask_valid <prompt> <default> <check>
  local prompt="$1" default="$2" check="$3" reply
  while :; do
    reply="$(ask "$prompt" "$default")"
    "$check" "$reply" && { printf '%s\n' "$reply"; return 0; }
    warn "$reply is not a usable value"
    interactive || die "bad value for $prompt: $reply"
  done
}

is_port()     { case "$1" in ''|*[!0-9]*) return 1;; esac; [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }
is_context()  { case "$1" in ''|*[!0-9]*) return 1;; esac; [ "$1" -ge 512 ]; }
is_fraction() { printf '%s' "$1" | grep -Eq '^(0?\.[0-9]+|1(\.0+)?)$'; }
yes_no()      { case "$1" in [Yy]|[Yy][Ee][Ss]) return 0;; *) return 1;; esac; }

expand() { printf '%s\n' "${1/#\~/$HOME}"; }

if [ "$FROM_CONFIG" -eq 0 ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    warn "no NVIDIA GPU detected: the published vLLM wheels are CUDA builds."
    warn "Point models.default at a server you run yourself instead —"
    warn "  hermes config --set models.default.base_url=http://localhost:11434/v1"
    die  "there is no local vLLM server for this script to start."
  fi
  [ -x "$VLLM_VENV/bin/vllm" ] \
    || die "no vLLM environment at $VLLM_VENV. Run ./install.sh first."

  # 8 GiB cards are the common case on laptops and the common cause of a
  # server that installs cleanly and then dies on first load.
  VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [ -n "$VRAM" ] && [ "$VRAM" -lt 10000 ] 2>/dev/null; then
    note "this GPU has ${VRAM} MiB of VRAM: serve at most a 7B 4-bit model."
    note "vLLM holds the whole model in VRAM, so a 30B model will not load."
  fi

  # Where the weights come from. Asked rather than guessed: an unattended run
  # that silently pulls several gigabytes is a worse outcome than one that
  # says which flag to pass.
  if [ -z "$MODEL_PATH" ] && [ -z "$MODEL_REPO" ]; then
    if interactive; then
      printf '\n'
      note "Which model should vLLM serve?"
      if [ -n "$CFG_MODEL_PATH" ]; then
        note "  1) keep serving $CFG_MODEL_PATH"
      else
        note "  1) download one from Hugging Face"
      fi
      note "  2) a model directory already on this machine"
      note "  3) download one from Hugging Face"
      case "$(ask "choice" 1)" in
        2) MODEL_PATH="$(expand "$(ask "path to the model directory" "$CFG_MODEL_PATH")")" ;;
        3) MODEL_REPO="$(ask "Hugging Face repository" "$DEFAULT_REPO")" ;;
        *) if [ -n "$CFG_MODEL_PATH" ]; then
             MODEL_PATH="$CFG_MODEL_PATH"
           else
             MODEL_REPO="$(ask "Hugging Face repository" "$DEFAULT_REPO")"
           fi ;;
      esac
    else
      MODEL_PATH="$CFG_MODEL_PATH"
      [ -n "$MODEL_PATH" ] || die \
        "no model chosen and no terminal to ask at. Pass --model REPO or
    --model-path DIR, or run this from a terminal."
    fi
  fi
  if [ -n "$MODEL_REPO" ] && [ -z "$MODELS_DIR" ]; then
    MODELS_DIR="$(expand "$(ask "store the weights in" \
      "$(dirname "${CFG_MODEL_PATH:-$DEFAULT_MODELS_DIR/x}")")")"
  fi

  PORT="$(ask_valid "port vLLM listens on" "${PORT:-$CFG_PORT}" is_port)"
  # One number, written to two places: vLLM reserves the KV cache for the full
  # context at startup, so a config asking for more than the server was started
  # with is a run that fails at the first long prompt.
  CONTEXT="$(ask_valid "context length to serve" "${CONTEXT:-$CFG_CONTEXT}" is_context)"
  GPU_UTIL="$(ask_valid "fraction of VRAM vLLM may use" "${GPU_UTIL:-$CFG_GPU_UTIL}" is_fraction)"

  if [ -x "$LAYA_VENV/bin/python" ] && [ "$CLASSIFIER" -eq -1 ]; then
    [ "$CFG_CLASSIFIER" -eq 1 ] && default=y || default=n
    yes_no "$(ask "also run the security-gate classifier? (y/n)" "$default")" \
      && CLASSIFIER=1 || CLASSIFIER=0
  fi
fi

MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS_DIR}"
CLASSIFIER_PORT="${CLASSIFIER_PORT:-$CFG_CLASSIFIER_PORT}"
CLASSIFIER_REPO="${CLASSIFIER_REPO:-$CFG_CLASSIFIER_MODEL}"

# -- weights -----------------------------------------------------------------

if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH="$MODELS_DIR/$(basename "$MODEL_REPO")"
  if [ -f "$MODEL_PATH/config.json" ]; then
    note "$MODEL_REPO is already in $MODEL_PATH"
  else
    # hf draws a per-file progress bar with real byte counts, so it is left to
    # speak for itself rather than being wrapped in a spinner.
    note "downloading $MODEL_REPO into $MODEL_PATH (progress below)"
    mkdir -p "$MODEL_PATH" || die "cannot write to $MODEL_PATH"
    "$VLLM_VENV/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_PATH" \
      || die "could not download $MODEL_REPO. Check the name, or pass --model-path."
  fi
fi
MODEL_PATH="$(cd "$MODEL_PATH" 2>/dev/null && pwd)" \
  || die "not a directory: $MODEL_PATH"
[ -f "$MODEL_PATH/config.json" ] \
  || warn "$MODEL_PATH has no config.json; vLLM may not recognise it"

# The name the config uses and the name vLLM answers to are made the same
# string on purpose: preflight compares what is configured against what
# /v1/models reports, and a path here would make that comparison fuzzy.
MODEL_NAME="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"

# -- the unit ----------------------------------------------------------------

# A systemd user unit rather than a shell command, because the factory runs for
# hours and every agent call needs the server up: something has to own
# restarts, and an operator who must remember to start it by hand is a
# dependency the loop cannot satisfy at 04:00. A *user* unit, so nothing here
# needs root and the weights stay readable by you alone.
#
# It is written but not enabled. vLLM pins the whole model in VRAM for as long
# as it lives, so a server running since boot is gigabytes of GPU given to an
# idle process; Hermes starts the unit when a project starts and stops it when
# the run ends. --boot opts into the always-up arrangement.
#
# Checked here rather than only where the questions are asked, because --config
# skips those: a unit whose ExecStart does not exist fails at start with a
# systemd error about a missing binary instead of the sentence you need.
[ -x "$VLLM_VENV/bin/vllm" ] \
  || die "no vLLM environment at $VLLM_VENV. Run ./install.sh first."
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
    --max-model-len $CONTEXT \\
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

# -- the config --------------------------------------------------------------

# Only the keys this script is the authority on, set in the file you already
# have: routing, budgets and every other choice in there are yours and are not
# touched. --config wrote nothing by definition, since its answers came from
# this file in the first place.
if [ "$FROM_CONFIG" -eq 0 ]; then
  note "updating $CONFIG"
  mkdir -p "$(dirname "$CONFIG")"
  # The classifier keys are written only when this run decided something about
  # it — a flag, or an answer to the question. Left at -1 they are not touched,
  # because a classifier you run yourself somewhere else is enabled in this
  # file by hand, and starting a model server is no reason to switch it off.
  CLASSIFIER_SETTINGS=()
  if [ "$CLASSIFIER" -ne -1 ]; then
    CLASSIFIER_SETTINGS=(
      --set "security.classifier.enabled=$([ "$CLASSIFIER" -eq 1 ] && echo true || echo false)"
      --set "security.classifier.base_url=http://localhost:$CLASSIFIER_PORT"
    )
  fi
  "$VENV/bin/hermes" -c "$CONFIG" config \
    --set "models.default.backend=vllm" \
    --set "models.default.base_url=http://localhost:$PORT/v1" \
    --set "models.default.model=$MODEL_NAME" \
    --set "models.default.context=$CONTEXT" \
    --set "server.unit=$UNIT_NAME" \
    --set "server.model_path=$MODEL_PATH" \
    --set "server.gpu_util=$GPU_UTIL" \
    "${CLASSIFIER_SETTINGS[@]}" \
    >/dev/null \
    || die "could not update $CONFIG"
fi

# -- the classifier ----------------------------------------------------------

# A second opinion on the security gate, on CPU deliberately: the coding model
# has the GPU for the whole run and vLLM pins it, so a 400M encoder competing
# for the same VRAM would cost more than the gate it serves is worth.
if [ "$CLASSIFIER" -eq 1 ]; then
  if [ ! -x "$LAYA_VENV/bin/python" ]; then
    warn "no classifier environment at $LAYA_VENV; run ./install.sh --classifier"
  else
    note "writing the $CLASSIFIER_UNIT user unit"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/$CLASSIFIER_UNIT" <<EOF
[Unit]
Description=Hermes Project Factory security classifier (port $CLASSIFIER_PORT)
After=network-online.target

[Service]
Type=exec
# The SDK deadlocks against TensorFlow's abseil init if both get imported.
Environment=USE_TF=0
ExecStart=$LAYA_VENV/bin/python $REPO/contrib/laya_server.py \\
    --model $CLASSIFIER_REPO \\
    --host 127.0.0.1 --port $CLASSIFIER_PORT \\
    --device cpu
Restart=always
RestartSec=10
TimeoutStartSec=600

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload || warn "could not reload the user systemd manager"
    if systemctl --user enable --now "$CLASSIFIER_UNIT" 2>/dev/null; then
      note "the classifier is running on port $CLASSIFIER_PORT"
    else
      warn "could not start $CLASSIFIER_UNIT; see: systemctl --user status $CLASSIFIER_UNIT"
    fi
    warn "$CLASSIFIER_REPO scores near chance unless it has been fine-tuned on"
    warn "labelled vulnerability data: leave security.classifier.block false"
    warn "until you have measured the checkpoint you are serving."
  fi
fi

# -- starting ----------------------------------------------------------------

if [ "$BOOT" -eq 1 ]; then
  # Lingering is the piece that makes "at boot" true rather than "at next
  # login", because a user unit otherwise stops with the last session.
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
  note "enabling $UNIT_NAME at boot and starting it now"
  systemctl --user enable "$UNIT_NAME" \
    || warn "could not enable $UNIT_NAME at boot"
fi

# restart rather than start when it is already up: the unit was just rewritten,
# and a running server is still serving the old model on the old port.
if systemctl --user is-active --quiet "$UNIT_NAME"; then
  note "restarting $UNIT_NAME with the settings above"
  systemctl --user restart "$UNIT_NAME" \
    || die "could not restart $UNIT_NAME; see: systemctl --user status $UNIT_NAME"
else
  note "starting $UNIT_NAME"
  systemctl --user start "$UNIT_NAME" \
    || die "could not start $UNIT_NAME; see: systemctl --user status $UNIT_NAME"
fi

# Loading several gigabytes onto the GPU takes minutes, and everything after
# this assumes the server answers, so wait rather than race it. The bar runs
# against the timeout rather than against real progress: vLLM reports nothing
# over HTTP until it is ready to serve, so elapsed-against-deadline is the only
# honest total available. It is labelled as such.
if [ "$WAIT" -eq 1 ]; then
  waited=0
  timeout="$CFG_WAIT"
  ready=0
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
    warn "the server did not answer on port $PORT within $((timeout / 60)) minutes."
    warn "check: systemctl --user status $UNIT_NAME; journalctl --user -u $UNIT_NAME"
    warn "a model too large for this GPU is the usual cause; --context and"
    warn "--gpu-util are the knobs, a smaller model is the other answer."
    exit 1
  fi
fi

# -- check -------------------------------------------------------------------

note "checking what Hermes sees"
"$VENV/bin/hermes" -c "$CONFIG" doctor \
  || warn "doctor found problems above. Hermes will not fall back to a cloud API."

cat <<EOF

==> serving $MODEL_NAME on http://localhost:$PORT/v1, $CONTEXT tokens of context.

    systemctl --user status $UNIT_NAME
    journalctl --user -u $UNIT_NAME -f    # what it is doing
    ./start.sh --stop                     # hand the VRAM back

The server is left running. Hermes only ever stops a server it started
itself, so this one stays up between runs; set 'stop_after_run: false' under
'server' if you want that of Hermes' own too.

Start a project from any directory:

    mkdir -p ~/projects/my-idea && cd ~/projects/my-idea
    hpf "my-idea"
EOF
