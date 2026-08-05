#!/bin/bash
# Build /tmp/app — a runnable shadow of the repo — and the bits Chromium needs.
#
# Three sandbox facts drive this file:
#   1. The sandbox ships Python 3.10; the app uses PEP 701 f-strings (3.12+).
#      Four files fail to parse.  `patch310.py` rewrites those four, and only
#      those four, in the shadow copy.  The repo itself is never touched.
#   2. Chromium is missing libXdamage.so.1 and there is no root to apt it in.
#      A four-symbol stub satisfies the loader; headless never calls into it.
#   3. Each bash call gets a fresh network namespace, so a server started in
#      one call is unreachable from the next — see session.py.
#
# Idempotent.  Run it before any scenario.
set -euo pipefail

REAL=/sessions/eloquent-stoic-turing/mnt/hbt-tools
APP=/tmp/app

# ── 1. libXdamage stub ───────────────────────────────────────────────────────
if [ ! -f /tmp/stublib/libXdamage.so.1 ]; then
  mkdir -p /tmp/stublib
  cat > /tmp/stublib/xd.c <<'EOF'
int XDamageQueryExtension(void*a,void*b,void*c){return 0;}
unsigned long XDamageCreate(void*a,unsigned long b,int c){return 0;}
void XDamageDestroy(void*a,unsigned long b){}
void XDamageSubtract(void*a,unsigned long b,unsigned long c,unsigned long d){}
EOF
  gcc -shared -fPIC -o /tmp/stublib/libXdamage.so.1 /tmp/stublib/xd.c
fi

# ── 2. shadow tree ───────────────────────────────────────────────────────────
# Code is copied (it gets patched).  Data and the guide folder are symlinked so
# uploads read the real files and screenshots land in the real guide/.
rm -rf "$APP"
mkdir -p "$APP"
cp -a "$REAL"/*.py "$APP"/ 2>/dev/null || true
cp -a "$REAL"/requirements.txt "$APP"/ 2>/dev/null || true
cp -a "$REAL"/.streamlit "$APP"/
cp -a "$REAL"/tools "$APP"/
find "$APP" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

for d in docs examples s2p dc_data deembed_these guide; do
  ln -sfn "$REAL/$d" "$APP/$d"
done
mkdir -p "$APP/dev"
ln -sfn "$REAL/dev/gds" "$APP/dev/gds"

# ── 3. downgrade the four 3.12-only files ────────────────────────────────────
python3 "$REAL/guide/_harness/patch310.py" "$APP"

# ── 3b. gdstk stand-in ───────────────────────────────────────────────────────
# No aarch64 wheel and no reachable Qhull 8 source, so the EBL page's GDS
# viewer would be switched off entirely.  The shim covers the five calls the
# app makes, on Shapely.  Sandbox only — it sits in the shadow tree's root so
# `import gdstk` finds it, and never enters the repo's import path.
python3 -c "import gdstk" 2>/dev/null \
  || cp "$REAL/guide/_harness/gdstk_shim.py" "$APP/gdstk.py"

# ── 4. sanity ────────────────────────────────────────────────────────────────
cd "$APP"
python3 - <<'EOF'
import pathlib, py_compile
bad = []
for p in pathlib.Path('tools').rglob('*.py'):
    try:
        py_compile.compile(str(p), doraise=True, cfile='/tmp/_x.pyc')
    except Exception:
        bad.append(str(p))
print("unparseable:", bad or "none")
EOF
echo "shadow ready at $APP"
