"""Visit the public dashboard in a headless browser, waking it if it sleeps.

Streamlit Community Cloud puts an idle app to sleep and serves a page with a
"Yes, get this app back up!" button instead of the app. A plain HTTP GET only
ever fetches that page, so waking it needs a real browser to click the button.

    python scripts/wake_app.py https://<your-app>.streamlit.app

Exit status is non-zero if the app never rendered, so the workflow goes red.
"""
from __future__ import annotations

import sys

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = "Yes, get this app back up!"


def main(url: str) -> int:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        button = page.get_by_text(WAKE_BUTTON)
        try:
            button.wait_for(timeout=15_000)
            print("app was asleep; clicking wake-up")
            button.click()
        except PWTimeout:
            print("no wake-up button; app already awake")
        # The running app lives in an iframe; its sidebar title proves it rendered.
        try:
            page.frame_locator("iframe").first.get_by_text("Tennis model").first.wait_for(
                timeout=240_000)
        except PWTimeout:
            print("app did not render within 4 minutes", file=sys.stderr)
            browser.close()
            return 1
        print("app is up")
        browser.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
