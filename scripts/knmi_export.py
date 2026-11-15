"""Write the Gate 1 (c) upload files for KNMI Climate Explorer.

    python scripts/knmi_export.py              # mexicali, iberia, amazon
    python scripts/knmi_export.py iberia       # one region

Files land in data/knmi/upload/; see docs/gate1.md for the upload steps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trendcourt import knmi  # noqa: E402

if __name__ == "__main__":
    for region in sys.argv[1:] or knmi.CASES:
        for path in knmi.export(region):
            print(path.relative_to(knmi.clerk.REPO_ROOT))
