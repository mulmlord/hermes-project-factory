"""Isolated execution of generated code.

Generated code is untrusted. The default runtime is a long-lived Docker
container with no network, capped CPU/RAM/PIDs, a read-only root with a
bounded ``/tmp``, and only the project workspace mounted. ``runtime: local`` exists for hosts without Docker and
is explicitly a weaker guarantee.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .log import Logger, NULL

#: The container runs as the host user, so pip cannot write to /usr/local.
#: PIP_USER sends every install to $HOME/.local instead, including installs
#: from project-written setup commands, and this PATH finds them again.
_PATH = "/tmp/.local/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin:/sbin"

#: Host paths that must never be visible to generated code.
FORBIDDEN_MOUNTS = ("~/.ssh", "~/.aws", "~/.config/gh", "~/.gnupg", "~/.kube")

#: Substrings that mark a variable name as a credential, matched against the
#: upper-cased name so ``github_token`` is caught as readily as ``GITHUB_TOKEN``.
#: Deliberately broader than the vendors of the day: the next key to appear on
#: a developer's machine will belong to a service nobody here has heard of, and
#: it will still be spelled with one of these words.
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASS", "CREDENTIAL", "AUTH",
                      "NETRC", "API_", "AWS_", "GH_", "GITHUB_", "ANTHROPIC",
                      "OPENAI")

#: Credentials whose names say nothing about what they are. A docker socket is
#: root on the host, and a kubeconfig is a path to cluster credentials.
#: :data:`FORBIDDEN_MOUNTS` hides the files these stand for; without this the
#: environment hands back the way in.
SECRET_ENV_NAMES = frozenset({"DOCKER_HOST", "DOCKER_CERT_PATH", "KUBECONFIG",
                              "GPG_AGENT_INFO"})

#: ``scheme://user:pass@host``. The one credential shape that survives every
#: check on the name, because the name is usually ``HTTPS_PROXY``.
_URL_CREDENTIALS = re.compile(r"([a-zA-Z][\w+.-]*://)[^/@\s]*:[^/@\s]*@")


def carries_secret(name: str) -> bool:
    """Whether generated code must not be told *name*.

    One predicate for every path that hands an environment to code Hermes did
    not write. It used to be two lists -- the sandbox's and the OpenHands
    runner's -- and the narrower of them passed ``HF_TOKEN`` and
    ``SSH_AUTH_SOCK`` through to an agent loop with a shell.
    """
    upper = name.upper()
    return upper in SECRET_ENV_NAMES or any(m in upper for m in SECRET_ENV_MARKERS)


def without_credentials(value: str) -> str:
    """Drop the ``user:pass@`` from any URL in *value*, leaving the URL usable.

    A proxy has to stay reachable for dependency installs, so the variable
    cannot simply be dropped -- but the password in it is a password.
    """
    return _URL_CREDENTIALS.sub(r"\1", value)


@dataclass
class Result:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self, limit: int = 4000) -> str:
        tail = (self.stdout + ("\n" + self.stderr if self.stderr else ""))[-limit:]
        status = "timeout" if self.timed_out else f"exit {self.exit_code}"
        return f"$ {self.command}\n[{status}, {self.seconds:.1f}s]\n{tail}"


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """Runs commands against a project workspace under resource limits."""

    def __init__(self, root: Path, cfg: dict, log: Logger = NULL):
        self.root = Path(root).resolve()
        self.cfg = cfg
        self.log = log
        self.runtime = cfg.get("runtime", "docker")
        self.workdir = cfg.get("workdir", "/workspace")
        #: A named OCI runtime -- ``runsc`` (gVisor), ``kata-runtime`` --
        #: for a kernel boundary a shared-kernel container does not have.
        #: A name rather than a free list of flags: a list would take
        #: ``--privileged`` as readily as ``--runtime``, and this knob exists
        #: only to add isolation.
        self.oci_runtime = str(cfg.get("oci_runtime") or "")
        self.container: str | None = None
        self.installed: list[str] = []  # tools Hermes had to add for the gates to run
        #: Seconds spent executing generated code. Paired with the router's
        #: own ledger it says whether a run was bound by the GPU or the CPU.
        self.spent = 0.0
        self._probe_error = ""
        #: Why the docker runtime was abandoned, empty while it still holds.
        #: Falling back to ``local`` weakens the isolation the config asked
        #: for, so the reason has to outlive the log line that announced it.
        self.fallback = ""
        if self.runtime == "docker" and not shutil.which("docker"):
            self._fall_back("sandbox.docker_missing", "docker is not installed")

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self.runtime != "docker" or self.container:
            return
        name = f"hermes-{self.root.name}-{uuid.uuid4().hex[:8]}"
        network = "none" if self.cfg.get("network", "none") != "host" else "bridge"
        cmd = ["docker", "run", "-d", "--rm", "--name", name,
               *self._confinement(network, self.workdir), "sleep", "infinity"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 and self.oci_runtime and self.oci_runtime in proc.stderr:
            # The hardened runtime is not registered with this engine. Plain
            # docker is still a container; ``local`` would be no box at all,
            # so step down one rung, not two, and say which one was lost.
            refused = self.oci_runtime
            self.oci_runtime = ""
            self.fallback = f"docker refused runtime {refused}: {proc.stderr.strip()[:200]}"
            self.log.warn("sandbox.runtime_refused", runtime=refused,
                          error=proc.stderr.strip()[:300], using="runc")
            cmd = ["docker", "run", "-d", "--rm", "--name", name,
                   *self._confinement(network, self.workdir), "sleep", "infinity"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            self._fall_back("sandbox.start_failed", proc.stderr.strip()[:300])
            return
        self.container = name
        if not self._workspace_visible():
            # Docker Desktop's WSL integration silently mounts an empty
            # directory when the host path also exists inside the engine VM
            # (/tmp is the usual victim). Gates would then all fail against a
            # workspace the container cannot see, so refuse to run blind.
            self.stop()
            self._fall_back(
                "sandbox.workspace_not_shared",
                f"the container cannot read {self.root}: {self._probe_error}",
                hint="the docker engine must share this path and the "
                     "directory must be readable inside the container")
            return
        self.log.info("sandbox.started", container=name, image=self.cfg.get("image"))

    def _fall_back(self, event: str, reason: str, **fields) -> None:
        """Give up on docker for this run, remembering what went wrong."""
        self.fallback = reason
        self.log.warn(event, error=reason, fallback="local", **fields)
        self.runtime = "local"

    def _cpu_limit(self) -> str:
        """The CPU cap, never more than the host has to give.

        Docker rejects a ``--cpus`` above the number of CPUs it can see, and
        it rejects the whole container with it. A config written for a bigger
        machine would then take every run down the fallback to ``local`` --
        trading the isolation away over a number that only ever meant "at
        most this much". Ask for less and the run is merely slower.
        """
        try:
            asked = float(self.cfg.get("cpus", 4))
        except (TypeError, ValueError):
            asked = 4.0
        capped = min(asked, float(os.cpu_count() or 1))
        if capped < asked:
            self.log.info("sandbox.cpus_capped", asked=asked, using=capped)
        return f"{capped:g}"

    @staticmethod
    def _user_args() -> list[str]:
        """Match the host user so build artifacts stay owned by the host."""
        if not hasattr(os, "getuid"):
            return []
        return ["--user", f"{os.getuid()}:{os.getgid()}"]

    def _workspace_visible(self) -> bool:
        """Confirm the bind mount really shows this workspace, not an empty dir."""
        token = uuid.uuid4().hex
        probe = self.root / f".hermes-mount-{token}"
        probe.write_text(token)
        try:
            seen = self.run(f"cat {shlex.quote(self.workdir)}/{probe.name}")
        finally:
            probe.unlink(missing_ok=True)
        self._probe_error = seen.stderr.strip()[:200]
        return seen.stdout.strip() == token

    def stop(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container],
                           capture_output=True, text=True)
            self.log.info("sandbox.stopped", container=self.container)
            self.container = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- execution -------------------------------------------------------
    def run(self, command: str, *, timeout: int | None = None, cwd: str | None = None,
            network: bool | None = None) -> Result:
        timeout = timeout or int(self.cfg.get("timeout", 900))
        started = time.time()
        if self.runtime == "docker":
            argv = self._docker_argv(command, cwd, network)
        else:
            argv = ["/bin/bash", "-lc", command]
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout,
                cwd=str(self.root / (cwd or ".")) if self.runtime != "docker" else None,
                env=self._local_env() if self.runtime != "docker" else None,
            )
            result = Result(command, proc.returncode, proc.stdout, proc.stderr,
                            time.time() - started)
        except subprocess.TimeoutExpired as exc:
            result = Result(command, 124, _text(exc.stdout), _text(exc.stderr),
                            time.time() - started, timed_out=True)
        self.spent += result.seconds
        self.log.debug("sandbox.run", command=command[:120], exit=result.exit_code,
                       seconds=round(result.seconds, 1))
        return result

    def install(self, command: str, *, timeout: int | None = None) -> Result:
        """Run a dependency install with network, then restore isolation.

        Installs have to land in the long-lived container or they vanish with
        the one-shot box, so the container is briefly attached to the bridge
        network rather than replaced. Everything else still runs offline.
        """
        if not command.strip() or command.strip() == "true":
            return Result(command, 0, "", "", 0.0)
        if self.runtime != "docker" or not self.container:
            return self.run(command, timeout=timeout, network=True)
        isolated = self.cfg.get("network", "none") != "host"
        if isolated and not self._reconnect("none", "bridge"):
            return self.run(command, timeout=timeout, network=True)
        try:
            return self.run(command, timeout=timeout or 900)
        finally:
            if isolated:
                self._reconnect("bridge", "none")

    def _reconnect(self, old: str, new: str) -> bool:
        """Swap the container between docker networks. False if docker refused."""
        for verb, net in (("disconnect", old), ("connect", new)):
            proc = subprocess.run(["docker", "network", verb, net, self.container],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                self.log.warn("sandbox.network", action=f"{verb} {net}",
                              error=proc.stderr.strip()[:200])
                return False
        return True

    def _docker_argv(self, command: str, cwd: str | None, network: bool | None) -> list[str]:
        if self.container and not network:
            work = self.workdir if not cwd else f"{self.workdir}/{cwd}"
            return ["docker", "exec", "-w", work, self.container,
                    "/bin/sh", "-c", command]
        # One-shot container: used when the caller needs network (dependency
        # installs, research fetches) which the long-lived box does not have.
        work = self.workdir if not cwd else f"{self.workdir}/{cwd}"
        return ["docker", "run", "--rm",
                *self._confinement("bridge" if network else "none", work),
                "/bin/sh", "-c", command]

    def _confinement(self, network: str, work: str) -> list[str]:
        """Every ``docker run`` flag that bounds the box, ending in the image.

        One list for the long-lived container and the one-shot box alike, so
        a limit added to one cannot be forgotten on the other.

        The root filesystem is read-only and ``/tmp`` -- which is ``$HOME``,
        and so where ``pip install --user`` lands -- is a tmpfs capped at
        ``sandbox.tmp``. Without that, a write to ``/tmp`` goes to the host's
        overlay store with no ceiling at all. The tmpfs is ``exec`` because
        installed console scripts run from it; docker's default is not.
        tmpfs pages count against ``--memory``, and ``--memory-swap`` equal
        to it means the cap is RAM rather than RAM plus as much swap again.

        ``docker run`` passes no host environment unless a ``-e`` asks it to,
        so the four below are the whole of the container's environment and
        none of them is a credential. Isolating the code and then handing it
        the host's keys would be a box with the door left open; adding a
        ``-e`` here is the only way that could happen, which is why the list
        is short enough to read.
        """
        memory = str(self.cfg.get("memory", "8g"))
        flags = [
            "--network", network,
            "--memory", memory, "--memory-swap", memory,
            "--cpus", self._cpu_limit(),
            "--pids-limit", str(self.cfg.get("pids", 512)),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        ]
        if self.oci_runtime:
            flags += ["--runtime", self.oci_runtime]
        if self.cfg.get("read_only", True):
            flags += ["--read-only", "--tmpfs",
                      f"/tmp:rw,exec,nosuid,nodev,mode=1777,size={self.cfg.get('tmp', '2g')}"]
        return flags + [
            "-v", f"{self.root}:{self.workdir}",
            "-w", work,
            *self._user_args(),
            "-e", "HOME=/tmp", "-e", f"PYTHONPATH={self.workdir}/src",
            "-e", "PIP_USER=1", "-e", f"PATH={_PATH}",
            self.cfg.get("image", "python:3.11-slim"),
        ]

    def _local_env(self) -> dict:
        """Strip host secrets before handing the environment to generated code.

        A blocklist, where the container gets an allowlist: ``local`` runs
        real builds on the host, and a toolchain reads more of the environment
        than anyone can enumerate ahead of it. That makes this the weaker rung
        by construction, which is the trade ``runtime: local`` already names.
        """
        env = {k: without_credentials(v) for k, v in os.environ.items()
               if not carries_secret(k)}
        env["HOME"] = str(self.root)
        env["HERMES_SANDBOX"] = "local"
        env["PYTHONPATH"] = str(self.root / "src")
        return env

    def run_all(self, commands: list[str], *, timeout: int | None = None,
                stop_on_failure: bool = True) -> list[Result]:
        results = []
        for command in commands:
            result = self.run(command, timeout=timeout)
            results.append(result)
            if stop_on_failure and not result.ok:
                break
        return results

    # -- file access -----------------------------------------------------
    def write(self, relative: str, content: str) -> Path:
        """Write inside the workspace, refusing escapes via .. or absolute paths."""
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def read(self, relative: str, limit: int | None = None) -> str:
        target = self.resolve(relative)
        if not target.is_file():
            return ""
        text = target.read_text(errors="replace")
        return text if limit is None else text[:limit]

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"path escapes workspace: {relative}")
        return candidate


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def registered_runtimes() -> set[str] | None:
    """OCI runtimes the docker engine will accept by name; None if it won't say."""
    if not shutil.which("docker"):
        return None
    proc = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return set(json.loads(proc.stdout or "{}"))
    except ValueError:
        return None


def probe_resources() -> dict:
    """Best-effort view of host CPU/RAM/disk/GPU for the scheduler."""
    info = {"cpus": os.cpu_count() or 1}
    try:
        with open("/proc/meminfo") as fh:
            mem = {line.split(":")[0]: line.split()[1] for line in fh if ":" in line}
        info["ram_gb"] = round(int(mem.get("MemTotal", 0)) / 1024 / 1024, 1)
        info["ram_free_gb"] = round(int(mem.get("MemAvailable", 0)) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        usage = shutil.disk_usage(".")
        info["disk_free_gb"] = round(usage.free / 1e9, 1)
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        if proc.returncode == 0:
            info["gpus"] = [
                dict(zip(("name", "vram_mb", "vram_free_mb"),
                         [p.strip() for p in line.split(",")]))
                for line in proc.stdout.strip().splitlines() if line.strip()
            ]
    return info
