"""Fetch and locally cache every source file.

CLAUDE.md flags single-maintainer risk on TennisMyLife (the same risk class that
took the Sackmann repos offline), so every file we pull is written to
data/raw/ and never re-fetched unless it changed upstream or the caller forces
it. The local cache, not the website, is the working copy.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from pathlib import Path

import requests

from tennis.config import (
    ONGOING_FILES,
    RAW_TD,
    RAW_TML,
    START_YEAR,
    TD_BASE,
    RAW_ONGOING_ARCHIVE,
    TML_API,
    TML_DATA,
)

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "tennis-predictor/1.0 (research; contact via repo owner)"}
MANIFEST = RAW_TML / "_manifest.json"


def _get(url: str, *, retries: int = 3, timeout: int = 60) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            # A 4xx is the server's final answer (missing file, bad path);
            # retrying it only delays the failure. 429 is the exception.
            if 400 <= r.status_code < 500 and r.status_code != 429:
                raise RuntimeError(f"failed to fetch {url}: HTTP {r.status_code}")
            r.raise_for_status()
            return r.content
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _write_atomic(dest: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a crash never leaves a torn file
    in the cache -- the cache is the working copy, not the website."""
    tmp = dest.with_name(f".{dest.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def archive_ongoing(name: str, data: bytes) -> int:
    """Keep the latest snapshot of every tournament seen in an ongoing file.

    The ongoing files are overwritten in place, and a tournament leaves them
    before the maintainer folds it into the yearly file. Without a copy of its
    own it vanished from the database for that gap (and CLAUDE.md asks for
    every pulled CSV to be cached). One file per tournament, overwritten while
    the tournament is still listed and left alone once it is not.
    """
    import pandas as pd
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str)
    except Exception:  # noqa: BLE001 - an unreadable file is still cached raw
        return 0
    if df.empty or "tourney_id" not in df.columns:
        return 0
    out = RAW_ONGOING_ARCHIVE / Path(name).stem
    out.mkdir(parents=True, exist_ok=True)
    for tid, g in df.groupby("tourney_id"):
        buf = io.BytesIO()
        g.to_csv(buf, index=False)
        _write_atomic(out / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', str(tid))}.csv",
                      buf.getvalue())
    return df["tourney_id"].nunique()


# --------------------------------------------------------------------------
# TennisMyLife
# --------------------------------------------------------------------------
def list_remote_files() -> list[dict]:
    """Return the upstream file listing (name, url, size, mtime)."""
    return json.loads(_get(TML_API))["files"]


def _targets(files: list[dict]) -> list[dict]:
    """Yearly ATP main-tour, ATP challenger and WTA files, plus the ongoing ones.

    Qualifying files are listed upstream and remain out of scope. WTA carries
    the identical 50-column schema, so this stays a filter rather than a
    parallel code path.
    """
    out = []
    for f in files:
        name = f["name"]
        m = re.fullmatch(r"(\d{4})(_challenger|_wta)?\.csv", name)
        if m and int(m.group(1)) >= START_YEAR:
            out.append(f)
        elif name in ONGOING_FILES:
            out.append(f)
    return out


def sync_tennismylife(*, force: bool = False, ongoing_only: bool = False) -> list[Path]:
    """Download any file whose upstream mtime/size differs from our cache."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = _targets(list_remote_files())
    if ongoing_only:
        files = [f for f in files if f["name"] in ONGOING_FILES]

    written: list[Path] = []
    for f in files:
        name = f["name"]
        dest = RAW_TML / name
        sig = {"mtime": f.get("mtime"), "size": f.get("size")}
        # Ongoing files always refetch: they change through the day and the
        # upstream mtime is not reliably bumped on every append.
        stale = force or name in ONGOING_FILES or manifest.get(name) != sig or not dest.exists()
        if not stale:
            continue
        data = _get(f.get("url") or f"{TML_DATA}/{name}")
        _write_atomic(dest, data)
        if name in ONGOING_FILES:
            archive_ongoing(name, data)
        manifest[name] = sig
        written.append(dest)
        log.info("cached %s (%s bytes)", name, dest.stat().st_size)

    _write_atomic(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True).encode())
    return written


# --------------------------------------------------------------------------
# tennis-data.co.uk  (odds, for backtesting)
# --------------------------------------------------------------------------
def sync_tennisdata(years: range | None = None, *, force: bool = False) -> list[Path]:
    """Download the yearly ATP odds workbooks.

    The site serves .xls for older seasons and .xlsx for newer ones with no
    published cutover year, so we probe both. Completed past seasons are never
    refetched; the current season is refetched every run.
    """
    from datetime import date

    this_year = date.today().year
    years = years or range(2001, this_year + 1)
    written: list[Path] = []
    for y in years:
        existing = list(RAW_TD.glob(f"{y}.xls*"))
        if existing and not force and y != this_year:
            continue
        for ext in ("xlsx", "xls"):
            url = f"{TD_BASE}/{y}/{y}.{ext}"
            try:
                r = requests.get(url, headers=HEADERS, timeout=60)
            except Exception:  # noqa: BLE001
                continue
            # The server answers 300 (Multiple Choices) with an HTML body when
            # the extension is wrong, so a 200 + non-HTML body is the real test.
            if r.status_code == 200 and not r.content[:15].lstrip().startswith(b"<"):
                dest = RAW_TD / f"{y}.{ext}"
                _write_atomic(dest, r.content)
                written.append(dest)
                log.info("cached odds %s (%s bytes)", dest.name, len(r.content))
                break
        else:
            log.warning("no odds workbook found for %s", y)
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n1 = sync_tennismylife()
    n2 = sync_tennisdata()
    print(f"tennismylife: {len(n1)} files; tennis-data: {len(n2)} files")
