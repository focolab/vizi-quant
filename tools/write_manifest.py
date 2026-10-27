"""Generate vizi-plugin.json from the interface declared in code.

    .venv/bin/python tools/write_manifest.py          # write it
    .venv/bin/python tools/write_manifest.py --check  # exit 1 if it is stale

Not ``vizi_quant_plugin.py --manifest``: that writes this machine's absolute
interpreter into ``entry.command``.  The tracked file keeps the portable
``.venv/bin/python`` (the host maps it to ``.venv\\Scripts\\python.exe`` on
Windows), the convention vizi-segment uses, so a release unpacks and runs
from wherever it is put.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = [".venv/bin/python", "vizi_quant_plugin.py"]


def generate() -> str:
    sys.path.insert(0, str(ROOT))
    import vizi_quant.plugin  # noqa: F401 -- importing it runs the declarations
    from vizi_plugin_sdk import declare, manifest

    data = declare.build(ENTRY)
    manifest.parse_manifest(data, ROOT)  # the host's own parser, before anything is written
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    text = generate()
    path = ROOT / "vizi-plugin.json"
    if "--check" in sys.argv[1:]:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            print("vizi-plugin.json is stale: run tools/write_manifest.py", file=sys.stderr)
            return 1
        print("vizi-plugin.json is current")
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
