import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"


def add_third_party(name):
    """Make an official repo under third_party/ importable (we run Python with -I, so nothing is on sys.path by default)."""
    path = str(THIRD_PARTY / name)
    if path not in sys.path:
        sys.path.insert(0, path)
