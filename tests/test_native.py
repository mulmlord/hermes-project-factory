"""The functions Hermes copies instead of installing.

Each section here is a third-party tool the factory used to depend on, and the
behaviour of it the factory actually needs, checked against the copy. The
copies are judged by what the original did -- PyYAML's scalar rules, pytest's
exit codes and report shape -- because the code that reads their output was
written against the original.
"""
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import textwrap
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from xml.etree import ElementTree

from hermes import testkit, yamlite
from hermes.config import DEFAULTS, SAMPLE_CONFIG, set_config_value

REPO = Path(__file__).resolve().parent.parent


# -- yaml ------------------------------------------------------------------------

#: Plain scalars and what PyYAML's safe_load made of them. YAML 1.1, so a
#: config written before the reader existed still means what it meant.
SCALARS = {
    "16384": 16384, "-5": -5, "+5": 5, "0": 0, "1_000": 1000, "0755": 0o755,
    "0x1F": 31, "0b101": 5, "0.90": 0.9, "1.5e+3": 1500.0, "1e3": "1e3",
    ".inf": math.inf, "-.Inf": -math.inf, "~": None, "null": None, "": None,
    "true": True, "yes": True, "On": True, "off": False, "No": False,
    "FALSE": False, "8g": "8g", "30m": "30m", "3.10": 3.1, "1.2.3": "1.2.3",
    "http://localhost:6767/v1": "http://localhost:6767/v1", "-x": "-x",
    "a:b": "a:b", "qwen2.5": "qwen2.5",
}


def test_plain_scalars_resolve_as_pyyaml_resolved_them():
    for text, value in SCALARS.items():
        loaded = yamlite.load(f"key: {text}\n")["key"]
        assert loaded == value and type(loaded) is type(value), (text, loaded)


def test_the_sample_config_reads_as_the_structure_it_describes():
    config = yamlite.load(SAMPLE_CONFIG)
    assert config["models"]["default"]["base_url"] == "http://localhost:6767/v1"
    assert config["models"]["default"]["context"] == 16384
    assert config["server"]["speculative"] == "ngram"
    assert config["sandbox"]["oci_runtime"] == ""
    # A flow mapping broken over two lines, with a comment on each.
    assert config["execution"]["escalation"] == [
        {"candidates": 2, "files": 10}, {"role": "architect", "format": "whole"}]
    assert config["research"]["search"] == ""


def test_every_shape_the_config_uses_reads_back():
    text = textwrap.dedent("""\
        # a comment
        top:
          nested: {a: 1, b: [x, 'y z', "q\\tr"]}
          list:
          - one
          - two: 2
            three: 3
          - - deep
          empty_map: {}
          empty_list: []
          quoted: 'it''s'
          hash: 'a # not a comment'   # but this is
          block: |
            line one
            line two
          folded: >-
            folded
            text
        """)
    assert yamlite.load(text) == {"top": {
        "nested": {"a": 1, "b": ["x", "y z", "q\tr"]},
        "list": ["one", {"two": 2, "three": 3}, ["deep"]],
        "empty_map": {}, "empty_list": [], "quoted": "it's",
        "hash": "a # not a comment", "block": "line one\nline two\n",
        "folded": "folded text"}}


def test_what_the_reader_does_not_read_is_refused_by_line():
    for text in ("a: &anchor 1\n", "a: *alias\n", "a: !!str 1\n",
                 "a: 1\n---\nb: 2\n", "a:\n\tb: 1\n", "a: b: c\n", "a: [1, 2\n"):
        try:
            yamlite.load(text)
        except yamlite.YAMLError as exc:
            assert exc.line, text
        else:
            raise AssertionError(f"accepted {text!r}")


def test_a_dump_reads_back_to_what_was_dumped():
    awkward = {"strings": ["yes", "16384", "", "null", "a: b", "#x", "- x", "[x]",
                           " lead", "trail ", "it's", 'say "hi"', "line\nbreak",
                           "\u2028", "tab\there", "---", "=", "<<"],
               "numbers": [0, -1, 2.5, 1e20, math.inf], "flags": [True, False, None],
               "nested": {"deeper": {"deepest": [{"k": "v"}, []]}, "empty": {}}}
    for data in (awkward, DEFAULTS):
        assert yamlite.load(yamlite.dump(data)) == data
        assert yamlite.load(yamlite.flow(data)) == data


def test_a_value_written_into_a_config_line_stays_unquoted_when_it_can():
    """start.sh writes URLs and paths with --set; quoting them would be noise."""
    text = set_config_value("models:\n  default:\n    model: x\n",
                            "models.default.base_url", "http://localhost:6900/v1")
    assert "base_url: http://localhost:6900/v1" in text
    assert yamlite.flow("16384") == "'16384'"


# -- the test runner -------------------------------------------------------------

def _run(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(REPO),
           "HOME": str(root)}
    return subprocess.run([sys.executable, "-m", "hermes.testkit", *args], cwd=root,
                          capture_output=True, text=True, env=env, timeout=120)


def _write(root: Path, files: dict) -> None:
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))


def test_the_runner_reports_each_outcome_and_exits_as_pytest_did(tmp_path):
    _write(tmp_path, {"tests/test_kinds.py": """\
        import sys
        from hermes import testkit

        def test_passes():
            assert 1 + 1 == 2

        def test_fails():
            print("visible only because I failed")
            print("so is this", file=sys.stderr)
            left, right = [1, 2], [1, 3]
            assert left == right

        @testkit.mark.skip(reason="not today")
        def test_skipped():
            raise AssertionError("never run")

        @testkit.mark.xfail(reason="known")
        def test_expected_failure():
            assert False

        def test_skips_itself():
            testkit.skip("from inside")

        def test_quiet_pass():
            print("never shown")
        """})
    proc = _run(tmp_path, "-q", "-rs")
    assert proc.returncode == 1, proc.stdout
    out = proc.stdout
    assert "2 passed" in out.splitlines()[-1] and "1 failed" in out.splitlines()[-1]
    assert "2 skipped" in out.splitlines()[-1] and "1 xfailed" in out.splitlines()[-1]
    assert "visible only because I failed" in out and "so is this" in out
    assert "never shown" not in out
    # A bare assert explains itself with the values it compared.
    assert "left = [1, 2]" in out and "right = [1, 3]" in out
    assert "SKIPPED [1] tests/test_kinds.py:13: not today" in out
    assert "from inside" in out

    _write(tmp_path, {"tests/test_kinds.py": "def test_only():\n    pass\n"})
    assert _run(tmp_path, "-q").returncode == 0
    _write(tmp_path, {"tests/test_kinds.py": "X = 1\n"})
    assert _run(tmp_path, "-q").returncode == 5          # nothing collected


def test_fixtures_set_up_in_scope_order_and_tear_down_in_reverse(tmp_path):
    _write(tmp_path, {
        "tests/conftest.py": """\
            from hermes import testkit
            EVENTS = []

            @testkit.fixture(autouse=True, scope="session")
            def session_wide():
                EVENTS.append("session up")
                yield
                EVENTS.append("session down")
                print("EVENTS", EVENTS)

            @testkit.fixture
            def outer():
                EVENTS.append("outer up")
                yield "outer"
                EVENTS.append("outer down")
            """,
        "tests/test_order.py": """\
            from hermes import testkit
            from conftest import EVENTS

            @testkit.fixture
            def inner(outer):
                EVENTS.append("inner up")
                yield outer + "+inner"
                EVENTS.append("inner down")

            def test_first(inner):
                assert inner == "outer+inner"

            def test_second(inner, tmp_path):
                assert tmp_path.is_dir()
                assert EVENTS == ["session up", "outer up", "inner up", "inner down",
                                  "outer down", "outer up", "inner up"]
            """})
    proc = _run(tmp_path, "-q", "-s")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ("EVENTS ['session up', 'outer up', 'inner up', 'inner down', 'outer down', "
            "'outer up', 'inner up', 'inner down', 'outer down', 'session down']") in proc.stdout


def test_parametrize_monkeypatch_capsys_raises_and_approx(tmp_path):
    _write(tmp_path, {"tests/test_api.py": """\
        import os
        import json
        from hermes import testkit

        @testkit.mark.parametrize("a,b", [(1, 2), (2, 4)], ids=["one", "two"])
        def test_doubles(a, b):
            assert a * 2 == b

        @testkit.mark.parametrize("word", ["x", "y"])
        def test_word(word):
            assert word in "xy"

        def test_patch(monkeypatch, tmp_path):
            monkeypatch.setattr("json.dumps", lambda value: "patched")
            monkeypatch.setenv("TESTKIT_PROBE", "on")
            monkeypatch.chdir(tmp_path)
            assert json.dumps(1) == "patched" and os.environ["TESTKIT_PROBE"] == "on"

        def test_patch_was_undone():
            assert json.dumps(1) == "1" and "TESTKIT_PROBE" not in os.environ

        def test_capsys(capsys):
            print("hello")
            assert capsys.readouterr().out == "hello\\n"

        def test_raises():
            with testkit.raises(ValueError, match="bad .* value"):
                raise ValueError("a bad input value")
            with testkit.raises(testkit.Failed):
                with testkit.raises(KeyError):
                    pass
            assert 0.1 + 0.2 == testkit.approx(0.3)
            assert [1.0000001] == testkit.approx([1.0])
        """})
    proc = _run(tmp_path, "-v")
    assert proc.returncode == 0, proc.stdout
    for nodeid in ("tests/test_api.py::test_doubles[one] PASSED",
                   "tests/test_api.py::test_doubles[two] PASSED",
                   "tests/test_api.py::test_word[x] PASSED"):
        assert nodeid in proc.stdout
    selected = _run(tmp_path, "-v", "-k", "doubles and not two")
    assert "test_doubles[one]" in selected.stdout and "test_word" not in selected.stdout


def test_the_junit_report_is_what_the_held_out_counter_reads(tmp_path):
    """harness.passed_tests was written against pytest's report; it must keep
    crediting exactly the cases that passed."""
    from hermes import harness

    _write(tmp_path, {"hermes_holdout/test_lib.py": """\
        import testkit

        @testkit.mark.parametrize("n", [1, 2, 3])
        def test_case(n):
            assert n != 2

        def test_single():
            pass
        """})
    runner = tmp_path / "hermes_holdout" / "testkit.py"
    runner.write_text((REPO / "hermes" / "testkit.py").read_text())
    proc = subprocess.run([sys.executable, str(runner), "-q", "--noconftest",
                           "--junitxml", "hermes_holdout/junit.xml",
                           "hermes_holdout/test_lib.py"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = (tmp_path / "hermes_holdout" / "junit.xml").read_text()
    cases = ElementTree.fromstring(report).iter("testcase")
    assert {c.get("classname") for c in cases} == {"hermes_holdout.test_lib"}
    expected = harness.expected_tests(tmp_path / "hermes_holdout")
    assert harness.passed_tests(report, expected) == 3


def test_a_conftest_the_project_ships_cannot_reach_the_held_out_run(tmp_path):
    _write(tmp_path, {
        "conftest.py": "raise SystemExit('a project conftest was imported')\n",
        "checks/test_x.py": "def test_x():\n    pass\n"})
    assert _run(tmp_path, "--noconftest", "-q", "checks").returncode == 0
    assert _run(tmp_path, "-q", "checks").returncode == 1


def test_a_module_that_will_not_import_is_an_error_not_a_silence(tmp_path):
    _write(tmp_path, {"tests/test_broken.py": "import no_such_module_anywhere\n",
                      "tests/test_fine.py": "def test_fine():\n    pass\n"})
    proc = _run(tmp_path, "-q")
    assert proc.returncode == 1
    assert "no_such_module_anywhere" in proc.stdout and "1 passed" in proc.stdout


def test_the_runner_is_one_file_of_the_standard_library():
    """The harness copies this file alone into a sandbox with nothing else in it."""
    import ast

    tree = ast.parse((REPO / "hermes" / "testkit.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import cannot survive being copied"
            imported.add(node.module.split(".")[0])
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported


# -- packaging ---------------------------------------------------------------------

def test_the_build_backend_makes_a_wheel_pip_can_install_offline(tmp_path):
    """No setuptools: the wheel is built by the backend in this repository."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=120)
    python = venv / "bin" / "python"
    proc = subprocess.run([str(python), "-m", "pip", "install", "-q", "--no-index",
                           str(REPO)], capture_output=True, text=True, timeout=300,
                          cwd=tmp_path)
    if proc.returncode != 0 and "No module named pip" in proc.stderr:
        testkit.skip("this interpreter's venv has no pip")
    assert proc.returncode == 0, proc.stderr
    probe = subprocess.run(
        [str(python), "-c", "import hermes, hermes.harness as h; "
                            "print(h.RUNNER.is_file(), sorted(p.name for p in "
                            "h.SUITE_DIR.iterdir()))"],
        capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert probe.stdout.strip() == "True ['cronspec', 'stencil', 'tally']", probe.stderr
    assert (venv / "bin" / "hpf").is_file() and (venv / "bin" / "hermes").is_file()


def test_the_project_declares_no_dependencies():
    import ast
    sys.path.insert(0, str(REPO / "build_backend"))
    try:
        import hermes_backend
    finally:
        sys.path.pop(0)
    project = hermes_backend._project()
    assert project.get("dependencies", []) == []
    assert "optional-dependencies" not in project
    assert hermes_backend.get_requires_for_build_wheel() == []
    # Every module the package imports is Hermes' own or the standard library's.
    for path in (REPO / "hermes").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                assert top in sys.stdlib_module_names or top in ("hermes", "__future__"), \
                    f"{path.name} imports {name}"


def test_the_workflow_runs_the_suite_this_repository_ships():
    """The dependency rule over the one file that has no imports to walk.

    When the suite moved off pytest, the extra it had been installed from went
    with it -- the test above is what holds that -- while the workflow went on
    asking pip for ``.[dev]`` and then calling ``pytest``. pip reports a
    missing extra as a warning rather than an error, so the break surfaced a
    step later as a missing command, on main, in the only file in the
    repository no import walk reaches. So the same property is asserted over
    what CI runs: it installs nothing, and every module it names exists here.
    """
    workflow = yamlite.load((REPO / ".github/workflows/tests.yml").read_text())
    commands = [step["run"] for job in workflow["jobs"].values()
                for step in job["steps"] if "run" in step]
    assert commands, "the workflow runs nothing"

    named = []
    for command in commands:
        assert "pip install" not in command, command
        tokens = command.split()
        assert tokens[0] != "pytest", command
        if "-m" in tokens:
            named.append(tokens[tokens.index("-m") + 1])

    assert "hermes.testkit" in named, commands
    for module in named:
        assert importlib.util.find_spec(module), module


# -- weights download ----------------------------------------------------------------

@contextlib.contextmanager
def _hub(files: dict, listing=None):
    """A loopback stand-in for the Hub: a paged listing, and resolve answers
    that redirect to another host the way the real storage redirect does."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization"),
                         self.headers.get("Range")))
            port = self.server.server_address[1]
            if self.path.startswith("/api/models/o/r/tree/main"):
                entries = listing if listing is not None else [
                    {"type": "file", "path": name, "size": len(body),
                     "lfs": {"oid": hashlib.sha256(body).hexdigest(), "size": len(body)}}
                    for name, body in files.items()]
                second = "cursor=2" in self.path
                page = entries[1:] if second else entries[:1]
                self.send_response(200)
                if not second and len(entries) > 1:
                    self.send_header("Link", f'<http://127.0.0.1:{port}/api/models/o/r/'
                                             'tree/main?recursive=true&cursor=2>; rel="next"')
                payload = json.dumps(page + ([{"type": "directory", "path": "sub"}]
                                             if second else [])).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path.startswith("/o/r/resolve/main/"):
                name = self.path.split("/o/r/resolve/main/", 1)[1]
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{port}/cdn/{name}")
                self.end_headers()
            elif self.path.startswith("/cdn/"):
                body = files[urllib.parse.unquote(self.path[5:])]
                start = 0
                if self.headers.get("Range"):
                    start = int(self.headers["Range"].split("=")[1].rstrip("-"))
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                self.wfile.write(body[start:])
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_weights_arrive_whole_and_the_token_stays_with_the_hub(tmp_path):
    from hermes import weights

    files = {"config.json": b'{"model_type": "qwen2"}',
             "sub/model.safetensors": os.urandom(3 * 1024 * 1024 + 17)}
    with _hub(files) as (endpoint, seen):
        fetched = weights.download("o/r", tmp_path, endpoint=endpoint, token="secret",
                                   stream=open(os.devnull, "w"))
    assert sorted(fetched) == sorted(files)
    for name, body in files.items():
        assert (tmp_path / name).read_bytes() == body
    assert not list(tmp_path.rglob("*.part"))
    for path, auth, _ in seen:
        if path.startswith("/cdn/"):
            assert auth is None, "the token followed a redirect off the Hub"
        else:
            assert auth == "Bearer secret", path


def test_an_interrupted_download_resumes_where_it_stopped(tmp_path):
    from hermes import weights

    body = os.urandom(2 * 1024 * 1024)
    (tmp_path / "w.bin.part").write_bytes(body[:700000])
    with _hub({"w.bin": body}) as (endpoint, seen):
        weights.download("o/r", tmp_path, endpoint=endpoint, token="",
                         stream=open(os.devnull, "w"))
        assert ("/cdn/w.bin", None, "bytes=700000-") in seen
        # And a second run downloads nothing: the file is there and checks out.
        assert weights.download("o/r", tmp_path, endpoint=endpoint, token="",
                                stream=open(os.devnull, "w")) == []
    assert (tmp_path / "w.bin").read_bytes() == body


def test_a_file_that_does_not_match_its_hash_is_not_kept(tmp_path):
    from hermes import weights

    body = b"x" * 1000
    listing = [{"type": "file", "path": "w.bin", "size": 1000,
                "lfs": {"oid": "0" * 64, "size": 1000}}]
    with _hub({"w.bin": body}, listing) as (endpoint, _):
        with testkit.raises(weights.WeightsError, match="SHA-256"):
            weights.download("o/r", tmp_path, endpoint=endpoint, token="",
                             stream=open(os.devnull, "w"))
    assert not (tmp_path / "w.bin").exists() and not (tmp_path / "w.bin.part").exists()


def test_a_listing_cannot_write_outside_the_target(tmp_path):
    from hermes import weights

    for path in ("../escape", "/etc/passwd", "a/../../b", "a//b", "a\\b"):
        listing = [{"type": "file", "path": path, "size": 1}]
        with _hub({}, listing) as (endpoint, _):
            with testkit.raises(weights.WeightsError, match="leaves the target"):
                weights.listing("o/r", endpoint=endpoint)
    with testkit.raises(weights.WeightsError, match="owner/name"):
        weights.listing("../../etc", endpoint="http://127.0.0.1:9")


# -- github ----------------------------------------------------------------------

@contextlib.contextmanager
def _api(routes: dict):
    """A loopback GitHub API answering *routes*: (method, path) -> (status, body)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"null")
            seen.append((self.command, self.path, self.headers.get("Authorization"), body))
            status, payload = routes.get((self.command, self.path), (404, {"message": "Not Found"}))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PATCH = _answer

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()


def test_the_rest_client_speaks_for_the_token_and_reads_errors():
    from hermes import github

    routes = {("GET", "/user"): (200, {"login": "someone"}),
              ("POST", "/repos/up/stream/pulls"): (201, {"html_url": "https://x/pr/1",
                                                         "number": 1}),
              ("GET", "/repos/up/stream"): (200, {"permissions": {"push": False}})}
    with _api(routes) as (url, seen):
        client = github.Client("tok", api=url)
        assert client.login() == "someone"
        assert client.repository(github.Slug("no", "such")) is None     # 404
        assert not client.can_push(github.Slug("up", "stream"))
        pull = client.open_pull(github.Slug("up", "stream"), title="t", head="someone:b",
                                base="main", body="b")
        assert pull["number"] == 1
        with testkit.raises(github.GitHubError, match="HTTP 404"):
            client.edit_issue(github.Slug("up", "stream"), 9, state="closed")
    assert {auth for _, _, auth, _ in seen} == {"Bearer tok"}
    assert seen[3][3]["head"] == "someone:b" and seen[3][3]["base"] == "main"


def test_a_push_carries_the_token_in_its_environment_not_its_argv(monkeypatch, tmp_path):
    from hermes import github

    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["env"] = argv, kwargs.get("env")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("hermes.github.subprocess.run", run)
    monkeypatch.setenv("SOME_OTHER_TOKEN", "unrelated secret")
    github.push(tmp_path, "origin", "HEAD", "tok-123")
    assert not any("tok-123" in part for part in seen["argv"])
    header = seen["env"]["GIT_CONFIG_VALUE_0"]
    assert header.startswith("AUTHORIZATION: basic ")
    assert seen["env"]["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert "SOME_OTHER_TOKEN" not in seen["env"]     # the one credential it needs


def test_a_github_remote_is_read_in_either_spelling():
    from hermes.github import parse_remote

    for url in ("https://github.com/o/r.git", "https://github.com/o/r",
                "git@github.com:o/r.git", "ssh://git@github.com/o/r"):
        assert str(parse_remote(url)) == "o/r", url
    assert parse_remote("https://example.invalid/o/r.git") is None
