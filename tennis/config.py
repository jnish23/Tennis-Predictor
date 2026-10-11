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
# Overridable so the cloud jobs (and tests) can work in a scratch directory
# without touching the laptop's database.
DATA = Path(os.environ.get("TENNIS_DATA_DIR") or ROOT / "data")
RAW_TML = DATA / "raw" / "tennismylife"
RAW_TD = DATA / "raw" / "tennisdata"
RAW_ONGOING_ARCHIVE = DATA / "raw" / "tennismylife_ongoing"
# Every ongoing file as pulled, under a dated name. Kept apart from the
# per-tournament snapshots above: load.read_raw globs that directory.
RAW_ONGOING_DATED = DATA / "raw" / "tennismylife_dated"
ARTIFACTS = Path(os.environ.get("TENNIS_ARTIFACTS_DIR") or ROOT / "artifacts")
DB_PATH = DATA / "tennis.db"

for _p in (RAW_TML, RAW_TD, RAW_ONGOING_ARCHIVE, RAW_ONGOING_DATED, ARTIFACTS):
    _p.mkdir(parents=True, exist_ok=True)

# Scope: ATP (main tour + challenger) and WTA (main tour only -- the source
# publishes no WTA challenger/ITF tier), 2000 onward.
START_YEAR = 2000
TOURS = ("atp", "wta")

TML_API = "https://stats.tennismylife.org/api/data-files"
TML_DATA = "https://stats.tennismylife.org/data"
TD_BASE = "http://www.tennis-data.co.uk"

# The ongoing-tournament files, refreshed near real time by the maintainer.
ONGOING_FILES = {
    "ongoing_tourneys.csv": "main",
    "challenger_ongoing_tourneys.csv": "challenger",
    "wta_ongoing_tourneys.csv": "wta",
}

# --- Categorical standardisation -------------------------------------------
# The live files do NOT use the tourney_level encoding described in CLAUDE.md
# (which said A/G/D/F). Observed values are 250/500/M/G/D/A/O/F/C. Mapping to a
# stable internal vocabulary so main-tour and challenger files can be unioned.
LEVEL_MAP = {
    "G": "grand_slam",     # Grand Slam
    "M": "masters",        # Masters 1000
    "F": "finals",         # Tour Finals
    "1000": "masters",     # WTA 1000 -- the WTA files use the number where
                           # the ATP files use "M" for the same tier
    "500": "atp500",
    "250": "atp250",
    # The WTA has renamed its tiers twice, so a 27-year history speaks three
    # vocabularies for the same ladder. Mapped onto the tokens above by
    # equivalent standing, not by name:
    #   2000-2008  Tier I..V
    #   2009-2020  Premier Mandatory / Premier 5 / Premier / International
    #   2021-      1000 / 500 / 250
    # No code collides with an ATP one, so a single shared map is enough.
    "T1": "masters",       # Tier I  -> today's 1000
    "T2": "atp500",
    "T3": "atp250",
    "T4": "atp250",
    "T5": "atp250",
    "PM": "masters",       # Premier Mandatory
    "P5": "masters",       # Premier 5
    "P": "atp500",         # Premier
    "I": "atp250",         # International
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
