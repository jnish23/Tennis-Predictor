"""Cooperative write lock between the long backfill and the nightly job.

SQLite's own locking is the wrong tool for this. The odds backfill runs for
days, committing every 25 matches; the nightly job needs a few seconds of
exclusive access to rebuild the match table. Left to `busy_timeout` the nightly
job simply loses -- it waits, times out and fails, every night for a week, and
before `load_all` took the write lock up front it did worse than fail: it
deleted 200,918 rows and then could not re-insert them.

So the two processes cooperate explicitly. The nightly job *claims* the
database; the backfill checks for a claim at its batch boundary -- a natural
yield point, since it has just committed and holds nothing -- and waits. The
claim is a file rather than a table row precisely because it must be readable
while another process holds the database's write lock.

A claim is ignored if the process that made it is gone. Otherwise a crashed
nightly job would stall the backfill indefinitely, trading a loud failure for a
silent one.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import datetime, timezone

from tennis.config import DATA

log = logging.getLogger(__name__)

CLAIM = DATA / ".db-write-claim"
STALE_AFTER = 1800.0     # seconds; a claim older than this is assumed dead


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError):
        return False
    return True


def _active_claim() -> dict | None:
    """The current claim, or None if there is none / it is dead."""
    try:
        c = json.loads(CLAIM.read_text())
    except (FileNotFoundError, ValueError):
        return None
    if not _alive(c.get("pid", -1)):
        log.info("clearing claim from dead pid %s", c.get("pid"))
        CLAIM.unlink(missing_ok=True)
        return None
    if time.time() - c.get("at", 0) > STALE_AFTER:
        log.warning("clearing stale claim from pid %s", c.get("pid"))
        CLAIM.unlink(missing_ok=True)
        return None
    return c


@contextlib.contextmanager
def exclusive_write(what: str = "", *, wait: float = 300.0, poll: float = 2.0):
    """Claim the database for a short heavy write, waiting for any other claim.

    Long-running writers yield at their next batch boundary, so the wait is
    normally one batch. Claiming does not itself block SQLite -- it asks nicely
    -- so a writer that ignores `should_yield` is unaffected.
    """
    deadline = time.time() + wait
    while _active_claim() is not None and time.time() < deadline:
        time.sleep(poll)
    CLAIM.parent.mkdir(parents=True, exist_ok=True)
    CLAIM.write_text(json.dumps({
        "pid": os.getpid(), "what": what, "at": time.time(),
        "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    try:
        yield
    finally:
        CLAIM.unlink(missing_ok=True)


class AlreadyRunning(RuntimeError):
    """Raised when a job that must be a singleton is already running."""


@contextlib.contextmanager
def single_instance(name: str):
    """Refuse to start if another live process is already running this job.

    Distinct from `exclusive_write`, which coordinates *different* jobs that
    both need the database. This stops the *same* job running twice, which the
    scraper has no defence against on its own: every write it makes is
    idempotent by primary key, so a second copy corrupts nothing, it just
    silently re-fetches pages the first copy is already fetching and doubles
    the request rate against a site we are deliberately rate-limiting. Two
    `--details` runs overlapped for nine hours before anyone noticed.

    The pid file is cleared if the process that wrote it is gone, so a crash
    does not lock the job out permanently. There is no staleness timeout here
    on purpose: unlike a write claim, a backfill legitimately runs for days.
    """
    pf = DATA / f".running-{name}"
    try:
        c = json.loads(pf.read_text())
        if _alive(c.get("pid", -1)):
            raise AlreadyRunning(
                f"{name} is already running as pid {c['pid']} "
                f"(started {c.get('since', 'unknown')}). "
                f"Stop it first, or wait for it to finish."
            )
        log.info("clearing pid file for dead %s (pid %s)", name, c.get("pid"))
    except (FileNotFoundError, ValueError):
        pass
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps({
        "pid": os.getpid(), "name": name,
        "since": datetime.now(timezone.utc).isoformat(timespec="seconds")}))
    try:
        yield
    finally:
        # Only clear our own pid file; a race that let two through must not
        # have the loser delete the winner's claim on its way out.
        try:
            if json.loads(pf.read_text()).get("pid") == os.getpid():
                pf.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass


def should_yield() -> dict | None:
    """Whether a short writer is waiting. Cheap enough to call every batch."""
    c = _active_claim()
    return c if c and c.get("pid") != os.getpid() else None


def wait_for_clear(poll: float = 5.0, max_wait: float = 1800.0) -> float:
    """Block until no other process holds a claim. Returns seconds waited."""
    t0 = time.time()
    c = should_yield()
    if c:
        log.info("yielding to %s (pid %s)", c.get("what") or "another writer",
                 c.get("pid"))
    while should_yield() is not None and time.time() - t0 < max_wait:
        time.sleep(poll)
    waited = time.time() - t0
    if waited > poll:
        log.info("resuming after %.0fs", waited)
    return waited
