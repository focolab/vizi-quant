#!/usr/bin/env python3
"""Check that this plugin's ``.venv`` is ready for vizi.  Read-only; exits
nonzero on any failure (the SDK's docs/environments.md contract).

    .venv/bin/python scripts/doctor.py

vizi-quant declares no assets and has no GPU code, so the checks are the
environment, the SDK, the plugin's declarations and manifest, and one tiny
measurement and plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"pass  {name}" + (f": {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 -- every failure is reported, none stops the rest
        FAILED.append(name)
        print(f"FAIL  {name}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=["cpu"], default="cpu")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    marker = {}

    def interpreter():
        nonlocal marker
        if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            raise RuntimeError(f"run with this plugin's .venv, not {sys.executable}")
        marker = json.loads((ROOT / ".venv" / ".vizi-environment.json").read_text())
        if marker.get("profile") != args.profile:
            raise RuntimeError(f".venv was bootstrapped for profile {marker.get('profile')}")
        return f"{sys.executable} (Python {sys.version.split()[0]}), {marker['target']}"

    def lock():
        path = ROOT / marker["lock"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != marker["lock_sha256"]:
            raise RuntimeError(f"{path.name} changed since bootstrap")
        wrong = []
        for name, version in re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", path.read_text(), re.M):
            have = metadata.version(name)
            if have != version:
                wrong.append(f"{name} {have} != {version}")
        if wrong:
            raise RuntimeError("; ".join(wrong))
        return f"{path.name} sha256 {digest[:12]}…, every pin installed"

    def sdk():
        import vizi_plugin_sdk

        return f"vizi-plugin-sdk {vizi_plugin_sdk.__version__}"

    def plugin():
        import vizi_quant.plugin  # noqa: F401 -- the declarations run on import
        from vizi_plugin_sdk import declare, manifest

        sys.path.insert(0, str(ROOT / "tools"))
        import write_manifest

        built = json.loads(write_manifest.generate())
        shipped = json.loads((ROOT / "vizi-plugin.json").read_text())
        if built != shipped:
            raise RuntimeError("vizi-plugin.json does not match the code")
        manifest.parse_manifest(shipped, ROOT)
        engines = shipped["engines"]["vizi"]
        if engines.split(".")[0] != "1":
            raise RuntimeError(f"engines.vizi {engines} is not protocol 1.x")
        return f"{shipped['id']} {shipped['version']}, engines.vizi {engines}, {len(shipped['contributions']['commands'])} commands"

    def measure():
        import numpy as np

        from vizi_quant import stats

        mask = np.ones((2, 3, 4), bool)
        values = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        got = stats.intensity(stats.region_values(values, mask))
        if not np.isclose(got[0], 11.5):
            raise RuntimeError(f"mean {got[0]} != 11.5")
        return "statistics on a synthetic region"

    def draw():
        import numpy as np

        from vizi_quant import plot

        png, _ = plot.render([("r", "r", np.arange(5.0))], 5, 2)
        if not png.startswith(b"\x89PNG"):
            raise RuntimeError("not a PNG")
        return f"trace plot rendered ({len(png)} bytes)"

    check("interpreter", interpreter)
    if marker:
        check("lock", lock)
    check("SDK import", sdk)
    check("plugin and manifest", plugin)
    check("measurement", measure)
    check("plot", draw)
    print("doctor: ready" if not FAILED else f"doctor: NOT ready ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
