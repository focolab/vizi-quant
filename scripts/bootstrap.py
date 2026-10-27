#!/usr/bin/env python3
"""Create this plugin's ``.venv`` from a checked-in lock -- the only supported
setup step (the SDK's docs/environments.md contract).

    python3.13 scripts/bootstrap.py
    .venv/bin/python scripts/doctor.py

vizi runs it this way itself -- no arguments, the plugin folder as the working
directory, on a Python of the minor ``.python-version`` names.  By hand, run
it with the Python you want the plugin to use; it picks the lock for this
OS, architecture and Python minor, refuses a tuple it has no lock for, and
installs with hash checking and wheels only.  It never touches an existing
``.venv`` built from something else: delete that one first.  Standard library
only, since it runs before any environment exists.

The venv links to its interpreter on Linux and macOS and copies it on Windows
(``venv_options``).  The Python vizi provides is a python-build-standalone
build whose binary finds libpython at ``@executable_path/../lib``; a copied
``.venv/bin/python`` has no ``../lib`` beside it and aborts at load on
macOS.  Windows has no such reference, and
symlinks there need privileges a person may not have.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
MARKER = VENV / ".vizi-environment.json"
SUPPORTED_MINORS = ((3, 11), (3, 13))


def target(profile: str) -> str:
    os_name = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system())
    arch = {"x86_64": "x86_64", "AMD64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(platform.machine())
    if os_name is None or arch is None:
        sys.exit(f"bootstrap: unsupported platform {platform.system()} {platform.machine()}")
    return f"{os_name}-{arch}-py{sys.version_info.major}{sys.version_info.minor}-{profile}"


def venv_options(os_name: str = os.name) -> dict:
    """``venv.create``'s options: link the interpreter except on Windows."""
    return {"with_pip": True, "symlinks": os_name != "nt"}


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # vizi runs this script with no arguments, so the only profile is the default.
    ap.add_argument("--profile", choices=["cpu"], default="cpu", help="vizi-quant has only a CPU profile (the default)")
    return ap.parse_args(argv)


def main() -> int:
    args = parse()

    if sys.version_info[:2] not in SUPPORTED_MINORS:
        sys.exit(f"bootstrap: Python {platform.python_version()} is not supported; use "
                 + " or ".join(f"{a}.{b}" for a, b in SUPPORTED_MINORS))
    pinned = (ROOT / ".python-version").read_text().strip()
    if pinned.rsplit(".", 1)[0] == f"{sys.version_info.major}.{sys.version_info.minor}" and platform.python_version() != pinned:
        # .python-version names the default interpreter's exact patch.  The
        # locks are per minor and pin every package, so another patch of the
        # same minor installs the same wheels: said, not refused.
        print(f"bootstrap: note: the release was built on Python {pinned}; this is {platform.python_version()}")

    name = target(args.profile)
    lock = ROOT / "requirements" / "locks" / f"{name}.txt"
    if not lock.exists():
        available = sorted(p.stem for p in (ROOT / "requirements" / "locks").glob("*.txt"))
        sys.exit(f"bootstrap: no lock for {name}; this release supports: {', '.join(available)}")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    wanted = {"target": name, "profile": args.profile, "lock": f"requirements/locks/{lock.name}",
              "lock_sha256": digest, "python": platform.python_version()}

    if VENV.exists():
        try:
            have = json.loads(MARKER.read_text())
        except (OSError, ValueError):
            sys.exit(f"bootstrap: {VENV} exists but was not made by this script; delete it and run again")
        if {k: have.get(k) for k in ("target", "profile", "lock_sha256")} != {k: wanted[k] for k in ("target", "profile", "lock_sha256")}:
            sys.exit(f"bootstrap: {VENV} was built for {have.get('target')} from a different lock; delete it and run again")
        print(f"bootstrap: {VENV} already matches {name}; nothing to do")
        return 0

    print(f"bootstrap: creating {VENV} for {name}")
    venv.create(VENV, **venv_options())
    py = str(venv_python())
    subprocess.run([py, "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes",
                    "--only-binary=:all:", "--find-links", str(ROOT / "vendor"), "-r", str(lock)], check=True)
    MARKER.write_text(json.dumps(wanted, indent=1) + "\n")
    print(f"bootstrap: done; check it with: {venv_python().relative_to(ROOT)} scripts/doctor.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
