"""Paths and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=value lines, no dependency, no overwrite.

    Real environment variables always win, so CI and the shell can override the
    file. Values are never logged -- this file holds credentials.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW_TML = DATA / "raw" / "tennismylife"
RAW_TD = DATA / "raw" / "tennisdata"
ARTIFACTS = ROOT / "artifacts"
DB_PATH = DATA / "tennis.db"

for _p in (RAW_TML, RAW_TD, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# Scope: ATP only for now (main tour + challenger), 2000 onward.
START_YEAR = 2000
TOURS = ("atp",)

TML_API = "https://stats.tennismylife.org/api/data-files"
TML_DATA = "https://stats.tennismylife.org/data"
TD_BASE = "http://www.tennis-data.co.uk"

# The ongoing-tournament files, refreshed near real time by the maintainer.
ONGOING_FILES = {
    "ongoing_tourneys.csv": "main",
    "challenger_ongoing_tourneys.csv": "challenger",
}

# --- Categorical standardisation -------------------------------------------
# The live files do NOT use the tourney_level encoding described in CLAUDE.md
# (which said A/G/D/F). Observed values are 250/500/M/G/D/A/O/F/C. Mapping to a
# stable internal vocabulary so main-tour and challenger files can be unioned.
LEVEL_MAP = {
    "G": "grand_slam",     # Grand Slam
    "M": "masters",        # Masters 1000
    "F": "finals",         # Tour Finals
    "500": "atp500",
    "250": "atp250",
    "A": "atp250",         # legacy pre-2009 "ATP Tour" label
    "D": "davis_cup",
    "O": "olympics",
    "C": "challenger",
}
# Which levels count as "main tour" vs "challenger" for reporting breakouts.
CHALLENGER_LEVELS = {"challenger"}

SURFACE_MAP = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}

# Rounds ordered from earliest to latest; index doubles as a numeric feature.
ROUND_ORDER = [
    "Q1", "Q2", "Q3", "R128", "R64", "R32", "R16", "QF", "SF", "BR", "F", "RR",
]
ROUND_MAP = {
    "R128": "R128", "R64": "R64", "R32": "R32", "R16": "R16",
    "QF": "QF", "SF": "SF", "F": "F",
    "RR": "RR",    # round robin (Tour Finals, Davis Cup group play)
    "BR": "BR",    # bronze medal match (Olympics)
    "Q1": "Q1", "Q2": "Q2", "Q3": "Q3",
}
