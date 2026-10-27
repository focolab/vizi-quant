"""The plugin must start on Windows, where CI does not run: no module may
import a POSIX-only module at import time.  Each check runs in a fresh
interpreter with those modules blocked, as Windows would refuse them."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Standard-library modules that do not exist on Windows.  (``posix`` is not
#: blocked: the standard library imports it itself whenever ``os.name`` is
#: ``"posix"``, so Linux cannot stand in for its absence.)
POSIX_ONLY = ("resource", "fcntl", "termios", "tty", "pty", "pwd", "grp", "crypt", "syslog")


def _run(code: str) -> subprocess.CompletedProcess:
    blocked = "\n".join(f"sys.modules[{name!r}] = None" for name in POSIX_ONLY)
    script = f"import sys\n{blocked}\nsys.path.insert(0, {str(ROOT)!r})\n" + textwrap.dedent(code)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)


def test_every_module_imports_without_posix_only_modules():
    done = _run("""
        import importlib, pkgutil, vizi_quant
        for mod in pkgutil.iter_modules(vizi_quant.__path__):
            importlib.import_module(f"vizi_quant.{mod.name}")
        import vizi_quant_plugin  # the entry the host launches
        print("ok")
    """)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr


def test_the_performance_log_samples_without_posix_only_modules():
    done = _run("""
        from vizi_quant import perf
        line = perf.sample(None)
        assert line["cpu_s"] >= 0 and "rss_mb" in line
        print("ok")
    """)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr
