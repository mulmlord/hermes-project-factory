"""Optional OpenHands execution layer.

OpenHands is used for tickets that benefit from a full software-engineering
agent loop: repository exploration, multi-file edits, iterative debugging. It
runs against the same workspace and the same local model endpoint. When it is
not installed, Hermes falls back to its native implementer and says so.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .agents import ticket_block
from .log import Logger, NULL

#: Tickets where a full agent loop tends to beat one-shot generation.
PREFERRED_KINDS = {"refactor"}


class OpenHands:
    def __init__(self, cfg: dict, model_cfg, log: Logger = NULL):
        self.cfg = cfg or {}
        self.model = model_cfg
        self.log = log
        self.command = self.cfg.get("command", "openhands")
        self.enabled = bool(self.cfg.get("enabled"))

    def available(self) -> bool:
        return self.enabled and bool(shutil.which(self.command.split()[0]))

    def should_handle(self, ticket) -> bool:
        if not self.available():
            return False
        if ticket.kind in PREFERRED_KINDS:
            return True
        # Multi-file work with several acceptance criteria, or a ticket the
        # native implementer has already failed twice.
        return len(ticket.artifacts) >= 3 or ticket.attempts >= 2

    def _env(self) -> dict:
        """Point OpenHands at the local endpoint. No hosted API keys are set."""
        base = self.model.base_url.rstrip("/")
        if self.model.backend == "ollama":
            model, api_base = f"ollama/{self.model.model}", base
        else:
            model = f"openai/{self.model.model}"
            api_base = base if base.endswith("/v1") else base + "/v1"
        env = dict(os.environ)
        env.update({
            "LLM_MODEL": model,
            "LLM_BASE_URL": api_base,
            # Required, not cosmetic: under --override-with-envs OpenHands
            # aborts when LLM_API_KEY or LLM_MODEL is unset. Local servers
            # ignore the value itself.
            "LLM_API_KEY": "local",
            "LLM_NUM_RETRIES": "2",
            "SANDBOX_USER_ID": str(os.getuid()),
            "LOG_ALL_EVENTS": "false",
        })
        for secret in list(env):
            if any(k in secret.upper() for k in
                   ("ANTHROPIC", "OPENAI_API_KEY", "GEMINI", "OPENROUTER", "GH_TOKEN",
                    "GITHUB_TOKEN", "AWS_")):
                env.pop(secret, None)
        return env

    def run_ticket(self, root: Path, ticket, context: str,
                   timeout: int = 1800) -> dict:
        """Hand a ticket to OpenHands. Returns a result dict for the caller."""
        task = (
            f"{context}\n\n{ticket_block(ticket)}\n\n"
            "Implement this ticket completely in the current repository. Run the "
            "validation commands yourself and fix what fails. Do not stop until "
            "they pass. Do not ask questions."
        )
        # Observed against openhands 1.21, not inferred from documentation: the
        # task is seeded with -t, the working directory is the process's own cwd
        # rather than a flag, and --override-with-envs is not optional -- without
        # it the LLM_* variables below are ignored and OpenHands falls back to
        # its stored settings, whose default endpoint is a hosted proxy
        # (llm-proxy.app.all-hands.dev). Hermes may not reach a hosted API, so
        # the flag is what keeps the guarantee.
        argv = [*self.command.split(), "--headless", "--override-with-envs",
                "-t", task]
        self.log.info("openhands.start", ticket=ticket.id)
        try:
            proc = subprocess.run(argv, cwd=root, env=self._env(), capture_output=True,
                                  text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            self.log.warn("openhands.failed", ticket=ticket.id, error=str(exc)[:200])
            return {"ok": False, "output": str(exc)}
        ok = proc.returncode == 0
        self.log.info("openhands.done", ticket=ticket.id, ok=ok)
        (root / ".hermes" / "logs").mkdir(parents=True, exist_ok=True)
        (root / ".hermes" / "logs" / f"openhands-{ticket.id}.log").write_text(
            proc.stdout + "\n" + proc.stderr)
        return {"ok": ok, "output": (proc.stdout + proc.stderr)[-8000:]}
