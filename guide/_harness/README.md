# Capture harness

How the screenshots in this guide are made. Nothing here ships with the site —
it exists so the images can be regenerated after a UI change.

## Sandbox facts that shaped the design

| fact | consequence |
|---|---|
| Each bash call gets a fresh **network namespace**, and detached processes are killed when the call returns | server + browser + capture must live in **one process, one call, under ~40 s** |
| The sandbox ships **Python 3.10**; the app uses PEP 701 f-strings (3.12+) | four files are folded onto single lines in a **shadow copy** at `/tmp/app` — the repo is never written to |
| Chromium is missing `libXdamage.so.1` and there is no root | a four-symbol stub in `/tmp/stublib` satisfies the loader |
| No emoji font, no `gdstk` wheel for aarch64, no reachable Qhull 8 | Noto Color Emoji is unpacked from npm into `~/.fonts`; `gdstk_shim.py` covers the five gdstk calls the app makes, on Shapely |

## Once per sandbox

```bash
bash guide/_harness/bootstrap.sh          # builds /tmp/app + the stub
```

Python deps (`streamlit plotly scipy playwright shapely fonttools brotli`) and
`playwright install chromium` are already in place; reinstall the same list if
the sandbox is recycled. Emoji font:

```bash
cd /tmp && npm pack @fontsource/noto-color-emoji && tar xzf fontsource-*.tgz
python3 -c "
from fontTools.ttLib import TTFont; import glob, os
for f in glob.glob('package/files/*-400-normal.woff2'):
    t = TTFont(f); t.flavor = None
    t.save(os.path.expanduser('~/.fonts/') + os.path.basename(f)[:-6] + '.ttf')"
fc-cache -f
```

## Capturing

```bash
bash guide/_harness/run.sh <scenario>     # -> guide/_shots/<scenario>/
```

A scenario is `guide/_harness/scenarios/<name>.py` exposing `run(page, shot)`:

```python
ENTRY = "IOED_Tool_Web.py"          # or tools/ebeam/calculator.py for standalone

def run(page, shot):
    page.get_by_role("link", name="RF At a Glance").first.click()
    page.locator('input[type="file"]').first.set_input_files(FILES)
    from session import settle; settle(page)
    shot("01_uploaded", full_page=True)
```

`shot(name, full_page=False, clip=None, locator=None)` writes
`<outdir>/<name>.png` at device scale 2 and waits for Streamlit to go quiet
first. Import `settle` from `session` after any interaction that triggers a
rerun — Streamlit blanks the DOM between runs and a screenshot taken in that
gap comes out empty.

Progress lands in `_status.json` (`starting` / `running` / `done` / `failed`)
and `_log.txt`; the server's own output is in `_streamlit.log`.

## Annotating

```python
from annotate import Annotator
(Annotator("guide/_shots/rf/01.png", css_width=1600)
    .marker(547, 380, 1)                       # numbered disc
    .circle(760, 400, r=28)
    .arrow(from_=(980, 300), to=(800, 390))
    .label(1000, 290, "Drop open.s2p and short.s2p here")
    .crop(440, 90, 1160, 640)
    .save("guide/docs/en/assets/rf/upload.png"))
```

Coordinates are CSS pixels — the same numbers Playwright reports from
`bounding_box()`. Get them by printing locator boxes inside the scenario rather
than guessing from the image.

## Gotchas

- `st.selectbox` renders as a **react-aria combobox** in this Streamlit build —
  `role="combobox"` plus an `aria-label="Open"` button. The BaseWeb
  `[data-baseweb="select"]` selector does not match.
- `st.container(border=True)` is `data-testid="stLayoutWrapper"`, not
  `stVerticalBlockBorderWrapper`.
- The RAM badge and language toggle are `position: fixed`, so they bleed into
  any `locator.screenshot()` whose element scrolls underneath. Hide them with
  `page.evaluate()` CSS injection at the point of capture —
  `add_init_script` races against Streamlit's client-side reruns.
- The fit cache at `~/.hbt-tools` **survives between scenarios** and the app
  auto-saves to it on any real parameter change. A scenario that commits a
  tuning result (`🏆 Use best values`) leaves the next scenario starting from
  those tuned values, not the seed. Re-run `seed_fit_cache.py` between any
  scenario that writes back and any scenario that depends on the canonical
  starting model.
- Files in the mounted repo **cannot be deleted**, only overwritten. Reruns
  overwrite in place; `_log.txt` appends.
- Run scenarios from `/tmp/app`, never from the repo — the repo's source does
  not parse under Python 3.10.
- `/tmp/app/guide` is a symlink back to the real folder, so screenshots written
  under it land in the repo. Don't `resolve()` paths through it.
