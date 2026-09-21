"""Shared fixtures."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def needs_docker(reason: str) -> None:
    """Skip the calling test, unless docker was promised.

    The docker-backed tests are the only check on the container path, and in
    the summary line a skip reads exactly like a pass. On a developer machine
    with no daemon that is the right trade; in CI it means the path can rot
    unnoticed, so ``HERMES_REQUIRE_DOCKER`` turns every such skip into a
    failure and the run has to say the daemon went away.
    """
    if os.environ.get("HERMES_REQUIRE_DOCKER"):
        pytest.fail(f"HERMES_REQUIRE_DOCKER is set but {reason}")
    pytest.skip(reason)


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


@pytest.fixture(autouse=True, scope="session")
def _ledger_elsewhere():
    """Keep the suite out of the developer's own ledger.

    The ledger is deliberately outside every workspace -- it is what the
    factory carries between projects -- so a test that opens the default one
    would write this run's invented failures into the record the next real
    project learns from, and leave a file behind on a machine that only ran
    the tests.
    """
    with tempfile.TemporaryDirectory(prefix="hermes-ledger-") as directory:
        os.environ["HERMES_LEDGER"] = str(Path(directory) / "factory.db")
        try:
            yield
        finally:
            os.environ.pop("HERMES_LEDGER", None)
