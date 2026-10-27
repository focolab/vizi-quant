"""What vizi relies on when it sets the plugin up itself: a bare
``scripts/bootstrap.py``, a ``.python-version`` it can read, and a hash-locked
requirements file for every platform and Python the plugin claims."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCKS = ROOT / "requirements" / "locks"
PLATFORMS = ("linux-x86_64", "windows-x86_64", "macos-arm64")


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = _script("bootstrap")


def test_bootstrap_needs_no_arguments():
    assert bootstrap.parse([]).profile == "cpu"
    assert bootstrap.parse(["--profile", "cpu"]).profile == "cpu"
    with pytest.raises(SystemExit):
        bootstrap.parse(["--profile", "gpu"])


def test_the_venv_links_its_interpreter_except_on_windows(monkeypatch, tmp_path):
    """A copied interpreter from vizi's python-build-standalone Python cannot
    find libpython on macOS; Windows copies work."""
    assert bootstrap.venv_options("posix") == {"with_pip": True, "symlinks": True}
    assert bootstrap.venv_options("nt") == {"with_pip": True, "symlinks": False}
    # and main() passes exactly those options to venv.create
    calls = []
    monkeypatch.setattr(bootstrap, "VENV", tmp_path / ".venv")
    monkeypatch.setattr(bootstrap.venv, "create", lambda path, **kw: calls.append(kw) or (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(bootstrap.sys, "argv", ["bootstrap.py"])
    with pytest.raises(SystemExit):
        bootstrap.main()
    assert calls == [bootstrap.venv_options()]


def test_python_version_names_a_supported_minor():
    text = (ROOT / ".python-version").read_text().strip()
    assert re.fullmatch(r"\d+(\.\d+)+", text), "digits and dots only: vizi truncates it to major.minor"
    major, minor = (int(v) for v in text.split(".")[:2])
    assert (major, minor) in bootstrap.SUPPORTED_MINORS


def test_a_lock_for_every_platform_and_minor():
    want = {f"{p}-py{a}{b}-cpu.txt" for p in PLATFORMS for a, b in bootstrap.SUPPORTED_MINORS}
    assert {p.name for p in LOCKS.glob("*.txt")} == want


@pytest.mark.parametrize("lock", sorted(LOCKS.glob("*.txt")), ids=lambda p: p.stem)
def test_every_requirement_in_a_lock_is_pinned_and_hashed(lock):
    # One requirement per logical line: the continuation lines are its hashes.
    logical = [r for r in re.sub(r"\\\n", " ", lock.read_text()).splitlines() if r.strip() and not r.lstrip().startswith("#")]
    names = []
    for line in logical:
        match = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)\s", line)
        assert match, f"not a pin: {line[:60]}"
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", line), f"{match.group(1)} has no hash"
        names.append(match.group(1))
    assert {"vizi-plugin-sdk", "numpy", "matplotlib", "grpcio", "protobuf", "atomicshm"} <= set(names)
    assert len(names) == len(set(names))
