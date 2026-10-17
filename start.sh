#!/usr/bin/env bash
#
# Set up and start the model server the Hermes Project Factory talks to.
#
# ./install.sh puts the software on disk and stops there. This script is the
# half that makes a decision: it asks which model to serve, from where, on
# which port and with how much context, writes those answers into your config,
# writes the systemd user unit that serves them, and brings the server up.
#
# It is also where the factory goes live, so it asks the one question about
# the factory itself: how often it should look after itself -- search the web
# for tools whose function it could rebuild natively, and run a
# self-optimisation round on this machine -- and whether an improvement your
# hardware finds may be offered upstream as a pull request that credits you.
#
# Run it again whenever those answers change — every question is pre-filled
# with what your config says now, so it doubles as an editor for them. With
# --config it asks nothing at all and starts exactly what the config already
# describes, which is the form to use from a script or a login item.
#
# Usage: ./start.sh [--config] [--config-file PATH]
#                   [--model REPO] [--model-path DIR] [--models-dir DIR]
#                   [--port N] [--context N] [--gpu-util F]
#                   [--upkeep WHEN] [--contribute] [--no-contribute]
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
#   --upkeep WHEN      how often the factory improves itself: weekly,
#                      biweekly, monthly, <n>d or off
#   --contribute       offer general improvements upstream as pull requests
#   --no-contribute    keep them on this machine
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
CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml"
UNIT_NAME="hermes-vllm.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

FROM_CONFIG=0
BOOT=0
WAIT=1
STOP=0
CONTRIBUTE=-1          # -1: ask, or whatever the config says; 0/1: the flags

# Answers. Empty means "not given on the command line", which is what makes a
# question worth asking.
MODEL_REPO=""
MODEL_PATH=""
MODELS_DIR=""
PORT=""
CONTEXT=""
GPU_UTIL=""
UPKEEP=""

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
    --upkeep)          UPKEEP="${2:?--upkeep needs a schedule}"; shift ;;
    --contribute)      CONTRIBUTE=1 ;;
    --no-contribute)   CONTRIBUTE=0 ;;
    --boot)            BOOT=1 ;;
    --no-wait)         WAIT=0 ;;
    --stop)            STOP=1 ;;
    -h|--help)         sed -n '3,44p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

[ -x "$VENV/bin/hermes" ] || die "Hermes is not installed here. Run ./install.sh first."

# -- stopping ----------------------------------------------------------------

# The counterpart to the whole script, because the one thing you want after
# "start it and keep it resident" is a way to hand the VRAM back.
if [ "$STOP" -eq 1 ]; then
  if [ -f "$UNIT_DIR/$UNIT_NAME" ] && systemctl --user is-active --quiet "$UNIT_NAME"; then
    note "stopping $UNIT_NAME"
    systemctl --user stop "$UNIT_NAME" || warn "could not stop $UNIT_NAME"
  fi
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
schedule = config.get("upkeep.schedule")


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
emit("CFG_UPKEEP", "off" if schedule in (None, False, "") else schedule)
emit("CFG_CONTRIBUTE", int(bool(config.get("upkeep.contribute"))))
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
is_schedule() { "$VENV/bin/python" -c 'import sys; from hermes.upkeep import interval; interval(sys.argv[1])' "$1" 2>/dev/null; }

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

  # Going live: the one question about the factory rather than the server.
  # Asked here because this is the only script that asks anything; the factory
  # itself never may (specification section 2.2), and a schedule is a decision
  # about your machine's time, which is yours to make.
  if [ -z "$UPKEEP" ] && interactive; then
    printf '\n'
    note "How often should the factory look after itself?"
    note "  Each time, it searches the web for tools whose function it would gain"
    note "  from and rebuilds that function natively -- it never installs them --"
    note "  then runs a self-optimisation round on this machine, judged by its"
    note "  own test suite and a before/after 'hermes eval' on your GPU."
    note "  weekly, biweekly, monthly, a number of days such as 10d, or off."
    [ "$CFG_UPKEEP" = "off" ] && default=weekly || default="$CFG_UPKEEP"
    UPKEEP="$(ask_valid "upkeep schedule" "$default" is_schedule)"
  fi
  UPKEEP="${UPKEEP:-$CFG_UPKEEP}"
  if [ "$UPKEEP" != "off" ] && [ "$CONTRIBUTE" -eq -1 ] && interactive; then
    note "When a run finds an improvement that holds on the whole evaluation suite,"
    note "the factory can open a pull request for it upstream, under your GitHub"
    note "account, and add you to IMPROVERS.md for the hardware you contributed."
    note "Maintainers review it like any other change. Needs a GitHub token in"
    note "  ${XDG_CONFIG_HOME:-$HOME/.config}/hermes/github-token"
    # Yes the first time the factory goes live; afterwards, what you said last.
    if [ "$CFG_UPKEEP" = "off" ] || [ "$CFG_CONTRIBUTE" -eq 1 ]; then default=y; else default=n; fi
    yes_no "$(ask "open pull requests for improvements your hardware finds? (y/n)" "$default")" \
      && CONTRIBUTE=1 || CONTRIBUTE=0
  fi
fi

MODELS_DIR="${MODELS_DIR:-$DEFAULT_MODELS_DIR}"

# -- weights -----------------------------------------------------------------

if [ -z "$MODEL_PATH" ]; then
  MODEL_PATH="$MODELS_DIR/$(basename "$MODEL_REPO")"
  if [ -f "$MODEL_PATH/config.json" ]; then
    note "$MODEL_REPO is already in $MODEL_PATH"
  else
    # Hermes' own downloader (hermes/weights.py): it reports real byte counts,
    # resumes an interrupted transfer and checks every large file's SHA-256,
    # so it is left to speak for itself rather than being wrapped in a spinner.
    # A gated repository needs HF_TOKEN in the environment.
    note "downloading $MODEL_REPO into $MODEL_PATH (progress below)"
    mkdir -p "$MODEL_PATH" || die "cannot write to $MODEL_PATH"
    "$VENV/bin/python" -m hermes.weights "$MODEL_REPO" "$MODEL_PATH" \
      || die "could not download $MODEL_REPO. Check the name, or pass --model-path.
    Run the same command again to resume an interrupted download."
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

# How to serve, as opposed to what. Hermes decides these two from the config
# and from which vLLM is installed — see hermes.server.serve_flags for why
# this workload wants them — and they are asked for by version rather than
# assumed, because a flag an older vLLM does not know is a server that exits
# on startup instead of one that serves slightly slower.
VLLM_VERSION="$("$VLLM_VENV/bin/python" -c \
  'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)"
[ -n "$VLLM_VERSION" ] && note "vLLM $VLLM_VERSION" \
  || warn "cannot tell which vLLM is installed at $VLLM_VENV"
SERVE_FLAGS="$("$VENV/bin/python" - "$CONFIG" "$VLLM_VERSION" <<'PY'
import copy, sys
from pathlib import Path

from hermes.config import DEFAULTS, Config
from hermes.server import serve_flags

path = Path(sys.argv[1])
config = Config.load(path) if path.is_file() else Config(copy.deepcopy(DEFAULTS))
for flag in serve_flags(config, sys.argv[2]):
    print(f" \\\n    {flag}", end="")
PY
)" || die "could not work out how to serve"

# Written by a function rather than inline because it is written twice: once
# with the flags above, and again without them if the server turns out not to
# accept them.
write_unit() {  # write_unit <extra ExecStart lines>
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
    --gpu-memory-utilization $GPU_UTIL$1
# vLLM drains in-flight requests on SIGINT; SIGTERM kills mid-generation.
KillSignal=SIGINT
Restart=always
RestartSec=10
TimeoutStartSec=900

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload || warn "could not reload the user systemd manager"
}

write_unit "$SERVE_FLAGS"

# -- the config --------------------------------------------------------------

# Only the keys this script is the authority on, set in the file you already
# have: routing, budgets and every other choice in there are yours and are not
# touched. --config wrote nothing by definition, since its answers came from
# this file in the first place.
if [ "$FROM_CONFIG" -eq 0 ]; then
  note "updating $CONFIG"
  mkdir -p "$(dirname "$CONFIG")"
  # The upkeep keys are written only when this run decided something about
  # them -- a flag, or an answer. Left alone they are not touched, because a
  # schedule set with `hermes upkeep --schedule` is yours as much as one set
  # here, and restarting a model server is no reason to change it.
  UPKEEP_SETTINGS=()
  [ -n "$UPKEEP" ] && [ "$UPKEEP" != "$CFG_UPKEEP" ] \
    && UPKEEP_SETTINGS+=(--set "upkeep.schedule=$UPKEEP")
  [ "$CONTRIBUTE" -ne -1 ] \
    && UPKEEP_SETTINGS+=(--set "upkeep.contribute=$([ "$CONTRIBUTE" -eq 1 ] && echo true || echo false)")
  "$VENV/bin/hermes" -c "$CONFIG" config \
    --set "models.default.backend=vllm" \
    --set "models.default.base_url=http://localhost:$PORT/v1" \
    --set "models.default.model=$MODEL_NAME" \
    --set "models.default.context=$CONTEXT" \
    --set "server.unit=$UNIT_NAME" \
    --set "server.model_path=$MODEL_PATH" \
    --set "server.gpu_util=$GPU_UTIL" \
    "${UPKEEP_SETTINGS[@]}" \
    >/dev/null \
    || die "could not update $CONFIG"
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
# Always a restart, whatever state it is in: "start" on a unit systemd is
# already restarting on a loop would be honoured lazily, and the whole point
# of the second call below is that it runs the unit file just rewritten.
start_unit() {
  if systemctl --user is-active --quiet "$UNIT_NAME"; then
    note "restarting $UNIT_NAME with the settings above"
  else
    note "starting $UNIT_NAME"
  fi
  systemctl --user restart "$UNIT_NAME" \
    || die "could not start $UNIT_NAME; see: systemctl --user status $UNIT_NAME"
}

# A server that died over an argument it did not recognise, as distinct from
# one still loading or one too big for the card. Both halves are needed: vLLM
# prints its speculation settings on a perfectly healthy startup, so the word
# alone means nothing, and "error" alone would blame the flags for a model
# that simply does not fit.
rejected_flags() {
  journalctl --user -u "$UNIT_NAME" -n 300 --no-pager 2>/dev/null \
    | grep -Ei 'error|unrecognized|invalid|unsupported|not supported' \
    | grep -Eqi 'speculativ|prefix.caching'
}

restarts() { systemctl --user show -p NRestarts --value "$UNIT_NAME" 2>/dev/null || echo 0; }

# Loading several gigabytes onto the GPU takes minutes, and everything after
# this assumes the server answers, so wait rather than race it. The bar runs
# against the timeout rather than against real progress: vLLM reports nothing
# over HTTP until it is ready to serve, so elapsed-against-deadline is the only
# honest total available. It is labelled as such.
#
# The one thing worth not waiting ten minutes for is a server that is not
# loading at all: a rejected argument makes vLLM exit in seconds and systemd
# restart it forever, so a restart the log blames on our flags ends the wait
# there and then.
wait_ready() {
  local waited=0 timeout="$CFG_WAIT" before
  before="$(restarts)"
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
      bar_clear
      note "vLLM is serving $MODEL_NAME on http://localhost:$PORT/v1"
      return 0
    fi
    if [ "$(restarts)" != "$before" ] && rejected_flags; then
      bar_clear
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
    bar "$waited" "$timeout" "loading $MODEL_NAME (${waited}s of ${timeout}s)"
  done
  bar_clear
  return 1
}

loading_failed() {
  warn "the server did not answer on port $PORT within $((CFG_WAIT / 60)) minutes."
  warn "check: systemctl --user status $UNIT_NAME; journalctl --user -u $UNIT_NAME"
  warn "a model too large for this GPU is the usual cause; --context and"
  warn "--gpu-util are the knobs, a smaller model is the other answer."
  exit 1
}

start_unit

if [ "$WAIT" -eq 1 ]; then
  if ! wait_ready; then
    # The flags above are there to make it faster, so they are never allowed
    # to make it absent: a vLLM that will not serve with them serves without
    # them instead, and says which knob turns that into the default.
    if [ -n "$SERVE_FLAGS" ] && rejected_flags; then
      warn "this vLLM refused:$(printf '%s' "$SERVE_FLAGS" | tr -d '\\' | tr '\n' ' ')"
      warn "serving without them. To stop trying them at every start:"
      warn "  hermes config --set server.speculative=off"
      write_unit ""
      SERVE_FLAGS=""
      start_unit
      wait_ready || loading_failed
    else
      loading_failed
    fi
  fi
elif [ -n "$SERVE_FLAGS" ]; then
  warn "not waiting, so a server that refuses the flags above goes unnoticed"
  warn "here: journalctl --user -u $UNIT_NAME is where it would say so."
fi

# -- check -------------------------------------------------------------------

note "checking what Hermes sees"
# Having waited, the server is known to answer, so it is worth asking it the
# questions that only a live server can: whether it holds a reply to a schema,
# whether its window is the one the config sizes prompts to.
PROBE=""
[ "$WAIT" -eq 1 ] && PROBE="--probe"
"$VENV/bin/hermes" -c "$CONFIG" doctor $PROBE \
  || warn "doctor found problems above. Hermes will not fall back to a cloud API."

# -- upkeep ------------------------------------------------------------------

# The timer is matched to the schedule on every start, so --config brings a
# machine that lost its units (a new home directory, a reinstall) back to what
# the config says without asking anything.
UPKEEP="${UPKEEP:-$CFG_UPKEEP}"
[ "$FROM_CONFIG" -eq 1 ] && UPKEEP="$CFG_UPKEEP"     # --config writes nothing
if [ "$UPKEEP" = "off" ]; then
  [ -f "$UNIT_DIR/hermes-upkeep.timer" ] && "$VENV/bin/hermes" -c "$CONFIG" upkeep --remove-timer
else
  "$VENV/bin/hermes" -c "$CONFIG" upkeep --install-timer \
    || warn "could not install the upkeep timer; run: hermes upkeep --install-timer"
  WANT_CONTRIBUTE="$CONTRIBUTE"
  [ "$WANT_CONTRIBUTE" -eq -1 ] && WANT_CONTRIBUTE="$CFG_CONTRIBUTE"
  if [ "$WANT_CONTRIBUTE" -eq 1 ]; then
    "$VENV/bin/python" -c 'import sys; from hermes.github import find_token; sys.exit(0 if find_token() else 1)' \
      || warn "contributing needs a GitHub token able to fork and open pull requests in
    ${XDG_CONFIG_HOME:-$HOME/.config}/hermes/github-token (chmod 600). Until then
    improvements are measured and kept on a local branch."
  fi
fi

cat <<EOF

==> serving $MODEL_NAME on http://localhost:$PORT/v1, $CONTEXT tokens of context.

    systemctl --user status $UNIT_NAME
    journalctl --user -u $UNIT_NAME -f    # what it is doing
    ./start.sh --stop                     # hand the VRAM back

The server is left running. Hermes only ever stops a server it started
itself, so this one stays up between runs; set 'stop_after_run: false' under
'server' if you want that of Hermes' own too.

Upkeep: $UPKEEP. See what it has done with: hermes upkeep --status

Start a project from any directory:

    mkdir -p ~/projects/my-idea && cd ~/projects/my-idea
    hpf "my-idea"
EOF
