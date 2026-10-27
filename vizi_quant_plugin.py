#!/usr/bin/env python3
"""What vizi launches (``entry.command`` in vizi-plugin.json).

At the plugin's root so the ``vizi_quant`` package beside it imports without
being installed: a release is an unpacked folder plus a bootstrapped ``.venv``.
``--install`` / ``--manifest`` are the SDK's, for a development checkout.
"""

from vizi_quant.plugin import main

if __name__ == "__main__":
    main()
