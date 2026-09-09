"""Shared fixtures."""
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def shared_tmp():
    """A temp dir the docker engine can actually bind-mount.

    Docker Desktop's WSL integration mounts an empty directory for /tmp paths,
    so docker-backed tests need a workspace under $HOME instead.
    """
    path = Path(tempfile.mkdtemp(prefix="hermes-test-", dir=Path.home()))
    path.chmod(0o755)  # the container drops CAP_DAC_OVERRIDE, so 0700 is unreadable
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
