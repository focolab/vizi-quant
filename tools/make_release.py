"""Build the release zip: ``dist/<plugin id>-<version>.zip``.

    .venv/bin/python tools/make_release.py

The zip holds one folder, ``org.focolab.vizi_quant/`` (the plugin id), with
what a fresh machine needs and nothing else: the manifest, the plugin code,
the Python pin, the requirement inputs and locks, the vendored SDK wheel,
bootstrap and doctor, the README, LICENSE and NOTICE.  Only files git tracks are packed, so a
stray local file cannot ship; the manifest and every lock must be current.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (
    "vizi-plugin.json", "vizi_quant_plugin.py", "vizi_quant/", "pyproject.toml", ".python-version",
    "requirements/", "vendor/", "scripts/bootstrap.py", "scripts/doctor.py", "tools/write_manifest.py",
    "README.md", "LICENSE", "NOTICE",
)


def main() -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    import write_manifest

    if json.loads(write_manifest.generate()) != json.loads((ROOT / "vizi-plugin.json").read_text()):
        sys.exit("vizi-plugin.json is stale: run tools/write_manifest.py")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SHIPPED], cwd=ROOT, capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit(f"uncommitted changes in shipped files:\n{dirty}")
    files = subprocess.run(["git", "ls-files", "--", *SHIPPED], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    manifest = json.loads((ROOT / "vizi-plugin.json").read_text())
    top = manifest["id"]
    out = ROOT / "dist" / f"{top}-{manifest['version']}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            zf.write(ROOT / name, f"{top}/{name}")
    print(f"wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
