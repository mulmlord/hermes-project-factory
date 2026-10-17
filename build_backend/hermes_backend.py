"""The build backend that packages Hermes, without setuptools.

``pip install .`` asks the project's build backend for a wheel, and the backend
is itself a package pip downloads first: setuptools, until this file. Hermes is
one pure-Python package with a few data files, which is a zip with a fixed
layout, so building it needs the standard library and nothing else. This is
that layout written out -- PEP 517 for wheels and sdists, PEP 660 for editable
installs -- and it is all ``pyproject.toml`` names under ``[build-system]``.

It reads the ``[project]`` table of ``pyproject.toml`` for the name, version,
description, Python requirement and console scripts. ``tomllib`` does that on
3.11 and later; on 3.10 a reader for the handful of TOML forms that file uses
stands in, and refuses anything else rather than guess.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import tarfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "hermes"
#: What a wheel carries besides Python sources: the eval suite's requirements
#: documents and held-out tests, which hermes.harness reads at runtime.
DATA = ("suite/*/*",)
#: What an sdist carries besides the package: enough to build the wheel again.
SDIST_EXTRA = ("pyproject.toml", "README.md", "LICENSE", "build_backend/hermes_backend.py")


# -- pyproject -----------------------------------------------------------------

def _toml_value(text: str, where: str):
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return bytes(text[1:-1], "utf-8").decode("unicode_escape")
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip().rstrip(",")
        return [_toml_value(part, where) for part in _split_list(inner)] if inner else []
    if text in ("true", "false"):
        return text == "true"
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    raise ValueError(f"pyproject.toml {where}: unsupported value {text!r}")


def _split_list(inner: str) -> list[str]:
    parts, current, quote = [], "", ""
    for char in inner:
        if quote:
            current += char
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
            current += char
        elif char == ",":
            parts.append(current)
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current)
    return parts


def _read_toml(text: str) -> dict:
    try:
        import tomllib
    except ImportError:          # Python 3.10
        tomllib = None
    if tomllib is not None:
        return tomllib.loads(text)
    data: dict = {}
    table = data
    pending = ""
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(" #", 1)[0].strip() if not raw.strip().startswith("#") else ""
        if pending:
            pending += " " + line
            if pending.count("[") > pending.count("]"):
                continue
            key, _, value = pending.partition("=")
            table[key.strip().strip('"')] = _toml_value(value, f"line {number}")
            pending = ""
            continue
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            table = data
            for part in line.strip("[]").split("."):
                table = table.setdefault(part.strip().strip('"'), {})
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"pyproject.toml line {number}: not a key = value line")
        if value.count("[") > value.count("]"):
            pending = line
            continue
        table[key.strip().strip('"')] = _toml_value(value, f"line {number}")
    return data


def _project() -> dict:
    return _read_toml((ROOT / "pyproject.toml").read_text())["project"]


def _names() -> tuple[str, str, str]:
    project = _project()
    name = project["name"]
    version = project["version"]
    return name, version, re.sub(r"[-_.]+", "_", name).lower()


# -- wheel contents ------------------------------------------------------------

def _metadata() -> str:
    project = _project()
    lines = ["Metadata-Version: 2.1", f"Name: {project['name']}",
             f"Version: {project['version']}"]
    if project.get("description"):
        lines.append(f"Summary: {project['description']}")
    if project.get("requires-python"):
        lines.append(f"Requires-Python: {project['requires-python']}")
    for requirement in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {requirement}")
    readme = ROOT / "README.md"
    if readme.is_file():
        lines.append("Description-Content-Type: text/markdown")
        return "\n".join(lines) + "\n\n" + readme.read_text()
    return "\n".join(lines) + "\n"


def _entry_points() -> str:
    scripts = _project().get("scripts", {})
    if not scripts:
        return ""
    return "[console_scripts]\n" + "".join(f"{k} = {v}\n" for k, v in scripts.items())


def _wheel_file(tag: str = "py3-none-any") -> str:
    return ("Wheel-Version: 1.0\nGenerator: hermes_backend\n"
            f"Root-Is-Purelib: true\nTag: {tag}\n")


def _package_files() -> list[Path]:
    base = ROOT / PACKAGE
    files = {p for p in base.rglob("*.py") if "__pycache__" not in p.parts}
    for pattern in DATA:
        files.update(p for p in base.glob(pattern) if p.is_file()
                     and "__pycache__" not in p.parts)
    return sorted(files)


def _record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def _zipinfo(name: str) -> zipfile.ZipInfo:
    # A fixed timestamp keeps two builds of the same tree byte-identical.
    stamp = time.gmtime(int(os.environ.get("SOURCE_DATE_EPOCH", "315532800")))[:6]
    info = zipfile.ZipInfo(name, date_time=max(stamp, (1980, 1, 1, 0, 0, 0)))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _write_wheel(directory: str, payload: list[tuple[str, bytes]]) -> str:
    _, version, dist = _names()
    filename = f"{dist}-{version}-py3-none-any.whl"
    info = f"{dist}-{version}.dist-info"
    entries = list(payload) + [
        (f"{info}/METADATA", _metadata().encode()),
        (f"{info}/WHEEL", _wheel_file().encode()),
        (f"{info}/top_level.txt", f"{PACKAGE}\n".encode()),
    ]
    points = _entry_points()
    if points:
        entries.append((f"{info}/entry_points.txt", points.encode()))
    record = [_record_line(name, data) for name, data in entries]
    record.append(f"{info}/RECORD,,")
    entries.append((f"{info}/RECORD", ("\n".join(record) + "\n").encode()))
    target = Path(directory) / filename
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in entries:
            archive.writestr(_zipinfo(name), data)
    return filename


# -- PEP 517 / PEP 660 hooks -----------------------------------------------------

def get_requires_for_build_wheel(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_sdist(config_settings=None) -> list[str]:
    return []


def get_requires_for_build_editable(config_settings=None) -> list[str]:
    return []


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    payload = [(path.relative_to(ROOT).as_posix(), path.read_bytes())
               for path in _package_files()]
    return _write_wheel(wheel_directory, payload)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None) -> str:
    """A wheel whose only payload is a .pth file naming this checkout.

    Python adds the directory a .pth line names to sys.path at startup, so
    the installed ``hermes`` is the source tree itself: an edit takes effect
    on the next run, and the self-improvement gate's PYTHONPATH still wins
    over it, because PYTHONPATH entries come before site-packages.
    """
    _, _, dist = _names()
    return _write_wheel(wheel_directory, [(f"__editable__.{dist}.pth",
                                           (str(ROOT) + "\n").encode())])


def build_sdist(sdist_directory, config_settings=None) -> str:
    _, version, dist = _names()
    base = f"{dist}-{version}"
    filename = f"{base}.tar.gz"
    members = [p.relative_to(ROOT).as_posix() for p in _package_files()]
    members += [m for m in SDIST_EXTRA if (ROOT / m).is_file()]
    target = Path(sdist_directory) / filename
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        pkg_info = _metadata().encode()
        info = tarfile.TarInfo(f"{base}/PKG-INFO")
        info.size, info.mode, info.mtime = len(pkg_info), 0o644, 315532800
        archive.addfile(info, io.BytesIO(pkg_info))
        for member in sorted(members):
            data = (ROOT / member).read_bytes()
            info = tarfile.TarInfo(f"{base}/{member}")
            info.size, info.mode, info.mtime = len(data), 0o644, 315532800
            archive.addfile(info, io.BytesIO(data))
    return filename
