"""run_galaxy.py

Convenience entrypoint for GalaXY.

Run:
    python run_galaxy.py

This launches the Napari + Qt GUI.
"""

from __future__ import annotations

from galaxy.gui import main


if __name__ == "__main__":
    main()
