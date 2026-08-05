#!/bin/bash
# Run one capture scenario, in the foreground.
#
#   bash guide/_harness/run.sh <scenario-name> [outdir-name]
#
# Foreground on purpose: a detached process is killed the moment the bash call
# returns, so a scenario must finish inside the 45 s call ceiling.  In practice
# that is roomy — the server is up in ~1 s warm and a 7-file upload plus two
# screenshots lands around 14 s.  Keep a scenario under ~35 s of app time.
set -uo pipefail

REAL=/sessions/eloquent-stoic-turing/mnt/hbt-tools
NAME="$1"
OUT="${2:-$NAME}"

[ -d /tmp/app ] || bash "$REAL/guide/_harness/bootstrap.sh" >/dev/null

export LD_LIBRARY_PATH=/tmp/stublib:${LD_LIBRARY_PATH:-}
export HBT_APP_ROOT=/tmp/app
export PATH=$HOME/.local/bin:$PATH

cd /tmp/app
exec timeout 43 python3 /tmp/app/guide/_harness/session.py \
    "/tmp/app/guide/_harness/scenarios/$NAME.py" \
    "/tmp/app/guide/_shots/$OUT"
