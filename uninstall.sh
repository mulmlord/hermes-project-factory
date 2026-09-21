#!/usr/bin/env bash
#
# Uninstall the Hermes Project Factory.
#
# Undoes ./install.sh: stops and removes the vLLM model server unit, removes
# the factory and vLLM virtualenvs, the `hpf` and `hermes` commands, the
# editable-install leftovers in this repository, and the optional OpenHands
# agent layer. It leaves the sandbox image and the downloaded model weights
# alone unless you ask for them: the image is a stock one other containers
# may share, and the weights are gigabytes you would have to fetch again.
#
# Your config is kept, because the model routing in it is yours, but the
# endpoint of the server being removed is retired in place so the file stops
# describing a server that no longer exists. The original is saved as
# config.yaml.bak.
#
# It never touches generated projects. Those are ordinary directories you own.
#
# Usage: ./uninstall.sh [--config] [--docker-image] [--models] [--all]
#                       [--keep-openhands] [--keep-server]
#                       [--system] [--bin DIR] [--venv DIR] [--dry-run]
#
#   --config           delete ~/.config/hermes/config.yaml instead of
#                      retiring the endpoint inside it, and delete what the
#                      factory learned about its own retries
#   --docker-image     also delete the pre-pulled sandbox image
#   --models           also delete the downloaded model weights
#   --all              --config and --docker-image together (not --models)
#   --keep-openhands   leave the OpenHands agent layer installed
#   --keep-server      leave the vLLM server unit and environment installed
#   --system           unlink from /usr/local/bin instead of ~/.local/bin (sudo)
#   --bin DIR          unlink the commands from DIR
#   --venv DIR         remove the virtualenv in DIR (default: <repo>/.venv)
#   --dry-run          report what would be removed, remove nothing
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
OH_VENV="$REPO/.venv-openhands"
LAYA_VENV="$REPO/.venv-laya"
VLLM_VENV="$REPO/.venv-vllm"
BIN_DIR="$HOME/.local/bin"
UNIT_NAME="hermes-vllm.service"
CLASSIFIER_UNIT="hermes-classifier.service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
MODELS_DIR="${HERMES_MODELS_DIR:-$HOME/.local/share/hermes/models}"
CONFIG=0
IMAGE=0
MODELS=0
OPENHANDS=1
SERVER=1
DRY=0

note() { printf '==> %s\n' "$*"; }
warn() { printf ' !  %s\n' "$*" >&2; }
die()  { printf 'XX  %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --config)         CONFIG=1 ;;
    --docker-image)   IMAGE=1 ;;
    --models)         MODELS=1 ;;
    --all)            CONFIG=1; IMAGE=1 ;;
    --keep-openhands) OPENHANDS=0 ;;
    --keep-server)    SERVER=0 ;;
    --system)         BIN_DIR="/usr/local/bin" ;;
    --bin)            BIN_DIR="${2:?--bin needs a directory}"; shift ;;
    --venv)           VENV="${2:?--venv needs a directory}"; shift ;;
    --dry-run)        DRY=1 ;;
    -h|--help)        sed -n '3,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

[ "$DRY" -eq 1 ] && note "dry run: nothing will be removed"

# A single place where anything is actually deleted, so --dry-run cannot miss a
# path and sudo is reached for only when the directory really is not ours.
remove() {
  local path="$1"
  if [ "$DRY" -eq 1 ]; then
    note "would remove $path"
  elif rm -rf "$path" 2>/dev/null; then
    note "removed $path"
  elif sudo rm -rf "$path"; then
    note "removed $path (as root)"
  else
    warn "could not remove $path"
  fi
}

# -- commands ----------------------------------------------------------------

# Only links that point back into an environment this script is removing. A
# `hermes` on PATH from somewhere else belongs to someone else's install, and
# silently deleting it would be the worst thing an uninstaller can do.
unlink_command() {
  local name="$1" link="$BIN_DIR/$1" target
  if [ ! -e "$link" ] && [ ! -L "$link" ]; then
    return
  fi
  if [ ! -L "$link" ]; then
    warn "$link is not a symlink; leaving it alone"
    return
  fi
  target="$(readlink "$link")"
  case "$target" in
    "$VENV"/*|"$OH_VENV"/*) remove "$link" ;;
    *) if [ ! -e "$link" ]; then
         note "$link is a broken link into $target"
         remove "$link"
       else
         warn "$link points at $target, which install.sh did not create; leaving it"
       fi ;;
  esac
}

note "unlinking commands from $BIN_DIR"
unlink_command hpf
unlink_command hermes

# -- model server ------------------------------------------------------------

# Stopped before the environment it runs from is deleted, or systemd keeps
# restarting a unit whose ExecStart has just vanished.
#
# Lingering is deliberately left enabled where it is on: only `install.sh
# --boot` turns it on, but other user services of yours may depend on it by
# now, and silently switching it off would stop them at your next logout.
if [ "$SERVER" -eq 1 ]; then
  if [ -f "$UNIT_DIR/$UNIT_NAME" ]; then
    note "stopping and removing $UNIT_NAME"
    if [ "$DRY" -eq 1 ]; then
      note "would run: systemctl --user disable --now $UNIT_NAME"
    else
      systemctl --user disable --now "$UNIT_NAME" 2>/dev/null \
        || warn "could not stop $UNIT_NAME; removing the unit file anyway"
    fi
    remove "$UNIT_DIR/$UNIT_NAME"
    [ "$DRY" -eq 0 ] && systemctl --user daemon-reload 2>/dev/null || true
  else
    note "no $UNIT_NAME unit installed"
  fi

  if [ -f "$VLLM_VENV/pyvenv.cfg" ]; then
    note "removing the vLLM environment"
    remove "$VLLM_VENV"
  fi

  if command -v loginctl >/dev/null 2>&1 \
     && [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = "yes" ]; then
    note "leaving lingering enabled for $USER (disable: loginctl disable-linger $USER)"
  fi
fi

# -- virtualenv --------------------------------------------------------------

if [ -f "$VENV/pyvenv.cfg" ]; then
  note "removing the factory virtualenv"
  remove "$VENV"
elif [ -e "$VENV" ]; then
  warn "$VENV is not a virtualenv; leaving it alone"
else
  note "no virtualenv at $VENV"
fi

# `pip install -e .` writes this next to the sources; without it a later
# checkout of a different version can still be picked up by a stale path entry.
for egg in "$REPO"/*.egg-info; do
  [ -d "$egg" ] && remove "$egg"
done

# -- security classifier -----------------------------------------------------

# Only ever installed by `install.sh --classifier`, so on most machines this
# finds nothing and says so once.
if [ -f "$UNIT_DIR/$CLASSIFIER_UNIT" ]; then
  note "stopping and removing $CLASSIFIER_UNIT"
  if [ "$DRY" -eq 1 ]; then
    note "would run: systemctl --user disable --now $CLASSIFIER_UNIT"
  else
    systemctl --user disable --now "$CLASSIFIER_UNIT" 2>/dev/null \
      || warn "could not stop $CLASSIFIER_UNIT; removing the unit file anyway"
  fi
  remove "$UNIT_DIR/$CLASSIFIER_UNIT"
  [ "$DRY" -eq 0 ] && systemctl --user daemon-reload 2>/dev/null || true
fi

if [ -f "$LAYA_VENV/pyvenv.cfg" ]; then
  note "removing the security classifier environment"
  remove "$LAYA_VENV"
fi

# -- openhands ---------------------------------------------------------------

# install.sh installs it one of two ways, so uninstall has to undo both. The uv
# tool is shared with anything else on this machine that uses it, hence the
# note: removing it is what was asked for, but it is worth saying out loud.
if [ "$OPENHANDS" -eq 1 ]; then
  if [ -f "$OH_VENV/pyvenv.cfg" ]; then
    note "removing the OpenHands agent layer"
    unlink_command openhands
    remove "$OH_VENV"
  elif command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q '^openhands '; then
    note "removing the OpenHands agent layer installed by uv"
    if [ "$DRY" -eq 1 ]; then
      note "would run: uv tool uninstall openhands"
    else
      uv tool uninstall -q openhands || warn "uv could not uninstall openhands"
    fi
  elif command -v openhands >/dev/null 2>&1; then
    warn "openhands at $(command -v openhands) was not installed by install.sh; leaving it"
  fi
fi

# -- configuration -----------------------------------------------------------

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/hermes/config.yaml"
CONFIG_DIR="$(dirname "$CONFIG_FILE")"
if [ "$CONFIG" -eq 1 ]; then
  if [ -f "$CONFIG_FILE" ]; then
    remove "$CONFIG_FILE"
    # Only if nothing else of yours lives there.
    if [ "$DRY" -eq 0 ] && [ -d "$CONFIG_DIR" ]; then
      rmdir "$CONFIG_DIR" 2>/dev/null && note "removed $CONFIG_DIR" || true
    fi
  else
    note "no config at $CONFIG_FILE"
  fi
elif [ -f "$CONFIG_FILE" ]; then
  note "keeping your config at $CONFIG_FILE (--config removes it)"
  # A kept config that still names the server just deleted is worse than no
  # config: Hermes would report the endpoint as unreachable and advise
  # starting it, which is bad advice once the unit and the weights are gone.
  # So the endpoint is retired rather than left to lie. The model name is
  # emptied, which makes Hermes fail at load with "no model name/path
  # configured" — the accurate complaint — instead of blaming the network.
  #
  # Only when this run actually removed the server, and only for a config
  # pointing at a local vLLM. The original is kept beside it either way.
  if [ "$SERVER" -eq 1 ] \
     && grep -q '^[[:space:]]*backend:[[:space:]]*vllm' "$CONFIG_FILE" \
     && grep -qE '^[[:space:]]*base_url:.*(localhost|127\.0\.0\.1)' "$CONFIG_FILE"; then
    if [ "$DRY" -eq 1 ]; then
      note "would retire the vLLM endpoint in $CONFIG_FILE (backup: .bak)"
    elif cp "$CONFIG_FILE" "$CONFIG_FILE.bak" \
         && awk -v today="$(date -I 2>/dev/null || date +%Y-%m-%d)" '
           /^models:/ && !banner {
             print "# The local vLLM server this config pointed at was removed by"
             print "# uninstall.sh on " today ". Hermes is local-model-only, so name"
             print "# the model and endpoint of a server you run, or re-run"
             print "# install.sh to set one up again."
             banner = 1
           }
           !done_model && /^[[:space:]]*model:[[:space:]]/ {
             print "    model: \"\"                       # set me: no server is installed"
             done_model = 1
             next
           }
           !done_base && /^[[:space:]]*base_url:[[:space:]]/ {
             print $0 "   # dead: this server was uninstalled"
             done_base = 1
             next
           }
           { print }
         ' "$CONFIG_FILE.bak" > "$CONFIG_FILE"; then
      note "retired the vLLM endpoint in $CONFIG_FILE"
      note "the previous version is at $CONFIG_FILE.bak"
    else
      warn "could not rewrite $CONFIG_FILE; it still names the removed server"
    fi
  fi
fi

# -- learned ledger ----------------------------------------------------------

# What the factory learned about its own retries across every project it built.
# It is not configuration and not a cache, and rebuilding it means building
# those projects again -- so it goes with --config, which is the flag that
# already means "remove what is mine as well".
LEDGER_FILE="${HERMES_LEDGER:-${XDG_STATE_HOME:-$HOME/.local/state}/hermes/factory.db}"
LEDGER_DIR="$(dirname "$LEDGER_FILE")"
if [ -f "$LEDGER_FILE" ]; then
  if [ "$CONFIG" -eq 1 ]; then
    remove "$LEDGER_FILE"
    remove "$LEDGER_FILE-wal"
    remove "$LEDGER_FILE-shm"
    if [ "$DRY" -eq 0 ] && [ -d "$LEDGER_DIR" ]; then
      rmdir "$LEDGER_DIR" 2>/dev/null && note "removed $LEDGER_DIR" || true
    fi
  else
    note "keeping what the factory learned at $LEDGER_FILE (--config removes it)"
  fi
fi

# -- sandbox image -----------------------------------------------------------

# Read from the config if it is still there, because a project may have been
# built against an image other than the default one install.sh pulls.
if [ "$IMAGE" -eq 1 ]; then
  if docker info >/dev/null 2>&1; then
    SANDBOX_IMAGE="$(sed -n 's/^[[:space:]]*image:[[:space:]]*//p' "$CONFIG_FILE" 2>/dev/null | head -1)"
    SANDBOX_IMAGE="${SANDBOX_IMAGE:-python:3.11-slim}"
    if docker image inspect "$SANDBOX_IMAGE" >/dev/null 2>&1; then
      note "removing the sandbox image $SANDBOX_IMAGE"
      if [ "$DRY" -eq 1 ]; then
        note "would run: docker rmi $SANDBOX_IMAGE"
      else
        docker rmi "$SANDBOX_IMAGE" >/dev/null \
          || warn "could not remove $SANDBOX_IMAGE; something may still be using it"
      fi
    else
      note "sandbox image $SANDBOX_IMAGE is not present"
    fi
  else
    warn "docker is not usable by this user; leaving the sandbox image"
  fi
fi

# -- model weights -----------------------------------------------------------

# Not part of --all: re-downloading several gigabytes is a slow way to learn
# that a flag was broader than it looked.
if [ "$MODELS" -eq 1 ]; then
  if [ -d "$MODELS_DIR" ]; then
    note "removing downloaded model weights from $MODELS_DIR"
    remove "$MODELS_DIR"
  else
    note "no downloaded weights at $MODELS_DIR"
  fi
elif [ -d "$MODELS_DIR" ]; then
  note "keeping the model weights in $MODELS_DIR (--models removes them)"
fi

# -- what is left ------------------------------------------------------------

if command -v hermes >/dev/null 2>&1 && [ "$DRY" -eq 0 ]; then
  warn "a hermes command is still on PATH at $(command -v hermes)"
  warn "it came from somewhere other than this repository's install.sh"
fi

cat <<EOF

==> uninstalled.

Untouched, because install.sh never created them:

    the projects Hermes built, and their .hermes/ state
    this repository (delete the directory itself to finish)
EOF
[ "$CONFIG" -eq 0 ] && [ -f "$CONFIG_FILE" ] && printf '    %s\n' "$CONFIG_FILE"
exit 0
