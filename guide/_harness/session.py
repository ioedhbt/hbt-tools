"""Boot the portal headless, drive it with Playwright, tear it down.

Everything happens inside one process because each sandbox bash call gets its
own network namespace — a Streamlit server started in one call is unreachable
from the next.  So: server + browser + capture, one process, one namespace.

Usage
-----
    python3 guide/_harness/session.py <scenario.py> [outdir]

A scenario is a module exposing ``run(page, shot)`` where

* ``page`` is a Playwright page already pointed at the app, and
* ``shot(name, **kw)`` saves ``<outdir>/<name>.png``.

Run it detached (``setsid nohup ... &``) and poll ``<outdir>/_status.json``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

# The app is run from the shadow tree (see bootstrap.sh), never from the repo:
# the sandbox's Python is 3.10 and the repo needs 3.12+ to parse.  `guide` inside
# the shadow tree is a symlink back to the real folder, so screenshots written
# under it land in the repo.  Don't resolve() your way to the root — that would
# follow the symlink straight back out.
APP = Path(os.environ.get("HBT_APP_ROOT", "/tmp/app"))
REPO = APP
STUBLIB = "/tmp/stublib"


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_http(url: str, timeout: float = 90.0) -> bool:
    import urllib.error
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(1)
    return False


def load_scenario(path: Path):
    spec = importlib.util.spec_from_file_location("scenario", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scenario"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    scenario_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "guide" / "_shots"
    outdir.mkdir(parents=True, exist_ok=True)
    status = outdir / "_status.json"
    log = outdir / "_log.txt"

    def note(**kw):
        payload = {"scenario": scenario_path.name, "ts": time.time(), **kw}
        status.write_text(json.dumps(payload, indent=2))

    def say(msg):
        with log.open("a") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    note(state="starting")
    say(f"scenario={scenario_path}")

    entry = getattr(load_scenario(scenario_path), "ENTRY", "IOED_Tool_Web.py")
    port = free_port()
    env = dict(os.environ)
    env["HBT_LOCAL_LAUNCH"] = "1"
    env["LD_LIBRARY_PATH"] = STUBLIB + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    server = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", entry,
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(REPO), env=env,
        stdout=(outdir / "_streamlit.log").open("w"), stderr=subprocess.STDOUT,
    )

    url = f"http://127.0.0.1:{port}"
    try:
        if not wait_http(url):
            note(state="failed", error="server never came up")
            return 1
        say(f"server up on {port}")

        from playwright.sync_api import sync_playwright

        scenario = sys.modules["scenario"]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                args=["--no-sandbox", "--disable-dev-shm-usage",
                      "--force-device-scale-factor=2"],
            )
            ctx = browser.new_context(
                viewport={"width": 1600, "height": 1000},
                device_scale_factor=2,
            )
            page = ctx.new_page()
            page.set_default_timeout(45_000)

            shots: list[str] = []

            def shot(name, full_page=False, clip=None, locator=None):
                settle(page)
                dest = outdir / f"{name}.png"
                if locator is not None:
                    locator.screenshot(path=str(dest))
                else:
                    page.screenshot(path=str(dest), full_page=full_page, clip=clip)
                shots.append(name)
                say(f"shot {name}")
                note(state="running", shots=shots)
                return dest

            page.goto(url, wait_until="domcontentloaded")
            settle(page)
            scenario.run(page, shot)
            browser.close()

        note(state="done", shots=shots)
        say("done")
        return 0
    except Exception:
        say(traceback.format_exc())
        note(state="failed", error=traceback.format_exc()[-2000:])
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()


def settle(page, quiet_ms: int = 1200, timeout_ms: int = 120_000) -> None:
    """Wait until Streamlit stops re-rendering.

    Streamlit reruns the whole script on every interaction and the DOM is
    briefly empty in between, so "no status widget" alone is not enough — a
    click can be screenshotted during the gap and come out blank.  Require the
    main area to hold non-trivial content, with the status widget gone, for
    `quiet_ms` in a row.
    """
    page.wait_for_timeout(400)
    end = time.time() + timeout_ms / 1000
    stable_since = None
    while time.time() < end:
        try:
            busy = page.locator('[data-testid="stStatusWidget"]').count() > 0
            body = page.locator('[data-testid="stMain"]').inner_text(timeout=2000)
        except Exception:
            busy, body = True, ""
        ready = (not busy) and len(body.strip()) > 20
        if ready:
            if stable_since is None:
                stable_since = time.time()
            elif (time.time() - stable_since) * 1000 >= quiet_ms:
                return
        else:
            stable_since = None
        page.wait_for_timeout(250)


if __name__ == "__main__":
    raise SystemExit(main())
